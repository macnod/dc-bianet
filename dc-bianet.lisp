(in-package :dc-bianet)

;; TODO:
;;
;;   - Ensure that the create-environment function's
;;     project-directory and train-path parameters can be specified
;;     without trailing slashes.
;;   
;;   - infer-frames doesn't really operate on frames
;;     

(defparameter *magnitude-limit* 1e9)
(defparameter *precision-limit* 1e-9)
(defparameter *default-learning-rate* 0.02)
(defparameter *default-momentum* 0.1)
(defparameter *default-min-weight* -0.9)
(defparameter *default-max-weight* 0.9)
(defparameter *default-thread-count*
  (let ((count (cl-cpus:get-number-of-processors)))
    (cond ((< count 3) 1)
          ((< count 9) (1- count))
          (t (- count 2)))))

(defparameter *job-queue* nil) ;; mailbox
(defparameter *job-counter* 0)
(defparameter *job-counter-mutex* (make-mutex :name "job-counter"))
(defparameter *thread-pool* nil) ;; A simple list
(defparameter *gates* nil)
(defparameter *main-training-thread* nil)
(defparameter *training-in-progress-mutex* (make-mutex :name "training-in-progress"))
(defparameter *continue-training* nil)

(defparameter *home-folder* (join-paths (namestring (user-homedir-pathname))
                                        "common-lisp" "dc-bianet"))

(defparameter *log-folder* "/tmp/bianet-logs/")

(defparameter *environments* nil) ;; p-list of id -> environment

;; Current environment
(defparameter *environment* nil)
(defparameter *net* nil)
(defparameter *network-error* 0.0)
(defparameter *training-set* nil)
(defparameter *test-set* nil)

(ensure-directories-exist (format nil "~a~a" 
                                  *log-folder*
                                  (if (ppcre:scan "/$" *log-folder*) "" "/")))

(defun start-swank-server ()
  (loop for potential-port = 4005 then (1+ potential-port) 
     for tries from 1 to 5
     for port = (handler-case 
		    (swank:create-server :port potential-port :dont-close t)
		  (sb-bsd-sockets:socket-error () nil))
     until port
     finally (format t "~%Started swank server on port ~d~%" port)
       (return port)))

(start-swank-server)

(defun stop-swank-servers ()
  (loop for port from 4005 to 4010 do (swank:stop-server port)))

(defun thread-work ()
  (loop for (k v) = (receive-message *job-queue*)
     do (case k
          ((:fire :backprop) 
           (funcall v)
           (inc-job-count))
          (:open-gate
           (open-gate (elt *gates* v)))
          (:stop
           (return)))))

(defun start-thread-pool (thread-count)
  (stop-thread-pool)
  (setf *job-queue* (make-mailbox :name "job-queue"))
  (setf *job-counter* 0)
  (setf *thread-pool*
        (loop for a from 1 to thread-count collect
             (make-thread #'thread-work :name "thread-work"))))

(defun stop-thread-pool ()
  (when *thread-pool*
    (loop 
       for (k v) in (receive-pending-messages *job-queue*)
       when (equal k :open-gate) 
       do (send-message *job-queue* (list k v)))
    (loop for thread in *thread-pool*
       do (send-message *job-queue* '(:stop nil)))
    (loop for thread in *thread-pool* 
       when (thread-alive-p thread)
       do (join-thread thread))
    (receive-pending-messages *job-queue*)
    (setf *thread-pool* nil)))

(defun terminate-thread-pool ()
  (when *thread-pool*
    (loop for thread in *thread-pool* do (terminate-thread thread))))

(defun inc-job-count(&optional (n 1))
  (with-mutex (*job-counter-mutex*)
    (incf *job-counter* n)))

(defun get-job-count ()
  (with-mutex (*job-counter-mutex*)
    *job-counter*))

(defun make-random-weight-fn (&key (min -0.5) (max 0.5))
  (lambda (&key rstate
             global-index
             global-fraction
             layer-fraction
             neuron-fraction)
    (declare (ignore global-index global-fraction layer-fraction
                     neuron-fraction))
    (+ min (random (- max min) rstate))))

(defun make-progressive-weight-fn (&key (min -0.5) (max 0.5))
  (declare (single-float min max))
  (lambda (&key rstate
             global-index
             global-fraction
             layer-fraction
             neuron-fraction)
    (declare (ignore rstate global-index global-fraction neuron-fraction))
    (+ min (* layer-fraction (- max min)))))
  
(defun make-sinusoid-weight-fn (&key (min -0.5) (max 0.5))
  (declare (single-float min max))
  (lambda (&key rstate
             global-index
             global-fraction
             layer-fraction
             neuron-fraction)
    (declare (ignore rstate global-fraction layer-fraction neuron-fraction))
    (+ (* (/ (+ (sin (coerce global-index 'single-float)) 1) 2) (- max min)) min)))

(defun display-float (n)
  (read-from-string (format nil "~,4f" n)))

(defun bianet-id ()
  "Returns a unique ID in the form of a Common Lisp keyword."
  (let ((s (format nil "~a~32R~32R"
                   (gensym)
                   (get-universal-time)
                   (random 1000000000))))
    (intern s :keyword)))

(defun logistic (x)
  (declare (single-float x)
           (optimize (speed 3) (safety 0)))
  (cond ((> x (the single-float 16.64)) (the single-float 1.0))
        ((< x (the single-float -88.7)) (the single-float 0.0))
        ((< (the single-float (abs x)) (the single-float 1e-8)) (the single-float 0.5))
        (t (/ (the single-float 1.0) (the single-float (1+ (the single-float (exp (- x)))))))))

(defun logistic-slow (x)
  (cond ((> x 16.64) 1.0)
        ((< x -88.7) 0.0)
        ((< (abs x) 1e-8) 0.5)
        (t (/ 1 (1+ (exp (- x)))))))

(defun logistic-derivative (x)
  (declare (single-float x)
           (optimize (speed 3) (safety 0)))
  (* x (- (the single-float 1.0) x)))

(defun relu (x)
  (declare (single-float x)
           (optimize (speed 3) (safety 0)))
  (the single-float (max (the single-float 0.0) x)))

(defun relu-derivative (x)
  (declare (single-float x)
           (optimize (speed 3) (safety 0)))
  (if (<= x (the single-float 0.0)) 
      (the single-float 0.0) 
      (the single-float 1.0)))

(defun relu-leaky (x)
  (declare (single-float x)
           (optimize (speed 3) (safety 0)))
  (the single-float (max (the single-float 0.0) x)))

(defun relu-leaky-derivative (x)
  (if (<= x (the single-float 0.0))
      (the single-float 0.001) 
      (the single-float 1.0)))
                              
(defparameter *transfer-functions* 
  (list :logistic (list :function #'logistic 
                        :derivative #'logistic-derivative)
        :relu (list :function #'relu
                    :derivative #'relu-derivative)
        :relu-leaky (list :function #'relu-leaky
                          :derivative #'relu-leaky-derivative)))

(defparameter *transfer-function-keys*
  (loop for key in *transfer-functions* by #'cddr collect key))

(defclass t-cx ()
  ((source :reader source :initarg :source :type t-neuron
           :initform (error ":source required"))
   (target :reader target :initarg :target :type t-neuron
           :initform (error ":target required"))
   (weight :accessor weight :initarg :weight :initform 0.1 :type single-float)
   (weight-dlist :accessor weight-dlist :type dlist 
                 :initform (make-instance 'dlist))
   (learning-rate :accessor learning-rate :initarg :learning-rate 
                  :type single-float :initform 0.02)
   (momentum :accessor momentum :initarg :momentum :type single-float 
             :initform 0.1)
   (delta :accessor delta :initarg :delta :type single-float :initform 0.0)
   (weight-mtx :reader weight-mtx :initform (make-mutex))))

(defclass t-neuron ()
  ((id :accessor id :initarg :id :type keyword :initform (bianet-id))
   (name :accessor name :initarg :name :type string :initform "")
   (input :accessor input :type single-float :initform 0.0)
   (biased :accessor biased :initarg :biased :type boolean :initform nil)
   (transfer-key :accessor transfer-key :initarg :transfer-key 
                 :initform :logistic)
   (transfer-function :accessor transfer-function :type function)
   (transfer-derivative :accessor transfer-derivative :type function)
   (output :accessor output :type single-float :initform 0.0)
   (expected-output :accessor expected-output :type single-float :initform 0.0)
   (err :accessor err :type single-float :initform 0.0)
   (err-derivative :accessor err-derivative :type single-float :initform 0.0)
   (x-coor :accessor x-coor :type single-float :initform 0.0)
   (y-coor :accessor y-coor :type single-float :initform 0.0)
   (z-coor :accessor z-coor :type single-float :initform 0.0)
   (cx-dlist :accessor cx-dlist :type dlist :initform (make-instance 'dlist))
   (input-mtx :reader input-mtx :initform (make-mutex))
   (output-mtx :reader output-mtx :initform (make-mutex))
   (err-mtx :reader err-mtx :initform (make-mutex))
   (err-der-mtx :reader err-der-mtx :initform (make-mutex))))

(defmethod initialize-instance :after ((neuron t-neuron) &key)
  (when (not (name neuron))
    (setf (name neuron) (format nil "~(~a~)" (id neuron))))
  (when (biased neuron)
    (setf (input neuron) 1.0))
  (let ((transfer (getf *transfer-functions* (transfer-key neuron))))
    (setf (transfer-function neuron)
          (getf transfer :function)
          (transfer-derivative neuron)
          (getf transfer :derivative))))

(defmethod transfer ((neuron t-neuron))
  (let* ((input (input neuron))
         (output (funcall (transfer-function neuron) input))
         (biased (biased neuron)))
    (with-mutex ((output-mtx neuron))
      (setf (output neuron) output))
    (with-mutex ((input-mtx neuron))
      (setf (input neuron) (if (not biased) 0.0 input)))
    (with-mutex ((err-mtx neuron))
      (setf (err neuron) nil))))

(defclass t-net ()
  ((id :reader id :initarg :id :type keyword :initform (bianet-id))
   (layer-dlist :accessor layer-dlist :type dlist :initform (make-instance 'dlist))
   (log-file :accessor log-file :initarg :log-file :initform nil)
   (weights-file :accessor weights-file :initarg :weights-file :initform nil)
   (rstate :accessor rstate :initform (make-random-state))
   (initial-weight-function 
    :accessor initial-weight-function
    :initarg :initial-weight-function
    :initform (make-progressive-weight-fn :min -0.5 :max 0.5))
   (connect-function
    :accessor connect-function
    :initarg :connect-function
    :initform (error ":connect-function is required"))))

(defmethod reset-random-state ((net t-net))
  (setf (rstate net) (make-random-state)))

(defclass environment ()
  ((id
    :accessor id :initarg :id :type keyword :documentation
    "The ID of the environment. Allows you, for example, to reference the
environment using the ENVIRONMENT-BY-ID function.")
   (net 
    :accessor net :initarg :net :type t-net :documentation
    "The neural network associated with this environment.")
   (project-directory 
    :accessor project-directory :initarg :project-directory
    :type (or pathname string) :documentation
    "The top-level directory where the neural network's training and test
data is found. This directory can contain CSV files (train.csv and
test.csv, for example) or it can contain a project directory tree. See
the files module for more information about the project directory
structure.")
   (train-path
    :accessor train-path :initarg :train-path :initform "train"
    :type (or pathname string) :documentation
    "Training data.  This can be a CSV file name, such as train.csv, which
resides in PROJECT-DIRECTORY. Or, it can be a data-set directory (also
in PROJECT-DIRECTORY). For information about the structure of a
data-set directory, see the file module.")
   (test-path
    :accessor test-path :initarg :test-path :initform "test"
    :type (or pathname string) :documentation
    "Testing data.  This can be a CSV file name, such as test.csv, which
resides in PROJECT-DIRECTORY. Or, it can be a data-set directory (also
in PROJECT-DIRECTORY). For information about the structure of a
data-set directory, see the file module.")
   (output-labels
    :accessor output-labels :initarg :output-labels :type t-output-labels
    :documentation
    "Information about the labels that required to create the training
 and testing data, as well as for inference. See the labels module for
 more information about labels.")
   (training-set
    :accessor training-set :initarg :training-set :initform (vector) 
    :type vector :documentation
    "A sample set, which is a frames vector. A frame is a list containing 2
lists that, together, represent a training or testing example for the
neural network.  Here's the structure:

    (vector                             <-- sample set
      (list inputs expected-outputs))   <-- frame
      (list inputs expected-outputs))   <-- frame
      ⋮                                      ⋮
    )

The inputs list contains floating-point values that represent the
input. The length of the inputs list corresponds exactly to the number
of inputs in the neural network.

The expected-outputs list contains zeros (0.0), except for one value,
which is 1.0. The index of that exception corresponds to the index of
the output that we want the the neural network to activate with the
given inputs.")
   (test-set 
    :accessor test-set :initarg :test-set :initform nil :type list
    :documentation
    "The structure of this data is exactly like the structure of the
training-set data, except that this data is to determine the accuracy
of the neural network after it has been trained.")
   (training-error 
    :accessor training-error :type dlist :initform (make-instance 'dlist)
    :documentation
    "Contains the network error values, so that they can be graphed during
and after training. Each element of this dlist consists of the
following values: elapsed-time, presentation-number, and network
error. Elapsed time is the number of seconds since training began.
Presentation number is the the number of times the neural network has
seen an example from the training set.  If there are 10 examples total
in the training set, and the neural network has seen each 10 times,
then presentation-number will be 100.")
   (training-error-limit
    :accessor training-error-limit :type integer :initarg :training-error-limit
    :initform 5000 :documentation
    "The number of network error values to keep in TRAINING-ERROR. When the
environment reaches this number of values, the oldest values fall off
the list.")
   (plot-errors
    :accessor plot-errors :initarg :plot-errors :initform nil :documentation
    "A boolean value that, when true, indicates that the environment should
plot network error during training.")
   (fitness
    :accessor fitness :type list :initform nil :documentation
    "A percentage value indicating the fitness of the neural network after
training.  This value is calculated by performing inference against
TEST-SET and then comparing the inference results against the expected
outputs for each example. A fitness of 100% indicates that the neural
network is infallible against the test set, which usually indicates a
problem.")))

(defun create-png-environment
    (id
     project-directory
     &key 
       (image-width 28)
       (image-height 28)
       (hidden-layer-topology '(16))
       (train-path "train")
       (test-path "test")
       (make-current t)
       (cx-mode :full)
       (cx-params 12)
       (learning-rate *default-learning-rate*)
       (momentum *default-momentum*))
  (ensure-data-paths-exist project-directory train-path test-path)
  (let* ((train-path-abs (join-paths project-directory train-path))
         (test-path-abs (join-paths project-directory test-path))
         (output-labels (make-instance 'output-labels 
                                       :data-set-path train-path-abs))
         (transformation (lambda (f)
                           (png-to-inputs f image-width image-height)))
         (training-set (create-sample-set 
                        train-path-abs
                        (label-outputs output-labels)
                        transformation))
         (test-set (create-sample-set (join-paths project-directory test-path)
                                      (label-outputs output-labels)
                                      transformation))
         (topology (create-topology hidden-layer-topology training-set))
         (net (create-standard-net
               topology
               :id id
               :cx-mode cx-mode
               :cx-params cx-params
               :learning-rate learning-rate
               :momentum momentum))
         (environment (make-instance 'environment
                                     :id id
                                     :net net
                                     :project-directory project-directory
                                     :train-path train-path
                                     :test-path test-path
                                     :output-labels output-labels
                                     :training-set training-set
                                     :test-set test-set
                                     :plot-errors t)))
    (setf (getf *environments* id) environment)
    (when make-current (set-current-environment id))
    environment))

(defun create-topology (hidden-layer-topology training-set)
  (flatten
   (list (length (car (elt training-set 0)))
         hidden-layer-topology
         (length (second (elt training-set 0))))))

(defmethod add-training-error ((environment environment)
                               (elapsed-seconds integer)
                               (presentation integer)
                               (network-error float))
  (push-tail (training-error environment)
             (list elapsed-seconds presentation network-error))
  (when (> (len (training-error environment))
           (training-error-limit environment))
    (pop-head (training-error environment))))

(defmethod plot-training-error ((environment environment))
  (loop for node = (head (training-error environment)) then (next node)
     while node
     for (elapsed presentation network-error) = (value node)
     collect elapsed into elapsed-seconds
     collect network-error into network-error-list
     finally (funcall #'plot elapsed-seconds network-error-list)))

(defmethod default-log-file-name ((net t-net))
  (join-paths *log-folder*
              (format nil "~(~a~)-~{~a~^-~}.log"
                      (id net)
                      (simple-topology net))))

(defmethod default-weights-file-name ((net t-net))
  (format nil "~atmp/~(~a~)-~{~a~^-~}-weights.dat"
          (user-homedir-pathname)
          (id net)
          (simple-topology net)))

(defmethod simple-topology ((net t-net))
  (loop for layer-node = (head (layer-dlist net)) then (next layer-node)
     while layer-node
     collect (len (value layer-node))))

(defgeneric feedforward (thing)
  (:method ((net t-net))
    (loop
       for layer-node = (head (layer-dlist net)) then (next layer-node)
       while layer-node
       for layer-gate-index = 0 then (1+ layer-gate-index)
       for layer-gate = (elt *gates* layer-gate-index)
       do 
         (close-gate layer-gate)
         (feedforward (value layer-node))
         (send-message *job-queue* (list :open-gate layer-gate-index))
         (wait-on-gate layer-gate)))
  (:method ((layer dlist))
    (loop 
       for neuron-node = (head layer) then (next neuron-node)
       while neuron-node
       do (send-message *job-queue* 
                        (list :fire
                              (let ((neuron (value neuron-node)))
                                (lambda () (feedforward neuron)))))))
  (:method ((neuron t-neuron))
    (loop initially (transfer neuron)
       for cx-node = (head (cx-dlist neuron)) then (next cx-node)
       while cx-node
       do (feedforward (value cx-node))))
  (:method ((cx t-cx))
    (let ((target-input-impact (* (weight cx) (output (source cx))))
          (target-neuron (target cx)))
      (with-mutex ((input-mtx target-neuron))
        (incf (input target-neuron) target-input-impact)))))

(defgeneric backpropagate (thing)
  (:method ((net t-net))
    (loop
       for layer-node = (tail (layer-dlist net)) then (prev layer-node)
       while layer-node
       for layer-gate-index = 0 then (1+ layer-gate-index)
       for layer-gate = (elt *gates* layer-gate-index)
       do
         (close-gate layer-gate)
         (backpropagate (value layer-node))
         (send-message *job-queue* (list :open-gate layer-gate-index))
         (wait-on-gate layer-gate)))
  (:method ((layer dlist))
    (loop
       for neuron-node = (tail layer) then (prev neuron-node)
       while neuron-node 
       do (send-message *job-queue* 
                        (list :backprop
                              (let ((neuron (value neuron-node)))
                                (lambda () (backpropagate neuron)))))))
  (:method ((neuron t-neuron))
    (compute-neuron-error neuron)
    (adjust-neuron-cx-weights neuron)))

(defmethod compute-neuron-error ((neuron t-neuron))
  (let ((err (if (zerop (len (cx-dlist neuron)))
                 ;; This is an output neuron (no outgoing connections)
                 (- (expected-output neuron) (output neuron))
                 ;; This is an input-layer or hidden-layer neuron;  we need 
                 ;; to use the errors of downstream neurons to compute the
                 ;; error of this neuron
                 (loop
                    for cx-node = (head (cx-dlist neuron)) then (next cx-node)
                    while cx-node
                    for cx = (value cx-node)
                    summing (the single-float 
                                 (* (the single-float (weight cx))
                                    (the single-float (err-derivative (target cx)))))))))
    (with-mutex ((err-mtx neuron)) (setf (err neuron) err))
    (with-mutex ((err-der-mtx neuron))
      (setf (err-derivative neuron)
            (* (the single-float err)
               (the single-float (funcall (transfer-derivative neuron) 
                                          (output neuron))))))))

(defmethod adjust-neuron-cx-weights ((neuron t-neuron))
  (loop
     with cx-dlist = (cx-dlist neuron)
     for cx-node = (head cx-dlist) then (next cx-node)
     while cx-node do (adjust-cx-weight (value cx-node))))

(defmethod adjust-cx-weight ((cx t-cx))
  (let* ((delta (* (the single-float (learning-rate cx))
                   (the single-float (err-derivative (target cx)))
                   (the single-float (output (source cx)))))
         (new-weight (+ (the single-float (weight cx))
                        (the single-float delta)
                        (the single-float (* (momentum cx) (delta cx))))))
    (with-mutex ((weight-mtx cx))
      (setf (weight cx) new-weight))))

(defmethod apply-inputs ((net t-net) (input-values list))
  (loop with input-layer-node = (head (layer-dlist net))
     with input-layer = (value input-layer-node)
     for neuron-node = (head input-layer) then (next neuron-node)
     while neuron-node
     for neuron = (value neuron-node)
     for input-value in input-values
     do (with-mutex ((input-mtx neuron))
          (setf (input neuron) (the single-float input-value)))))

(defmethod apply-outputs ((net t-net) (output-values list))
  (loop with output-layer-node = (tail (layer-dlist net))
     with output-layer = (value output-layer-node)
     for neuron-node = (head output-layer) then (next neuron-node)
     while neuron-node
     for neuron = (value neuron-node)
     for output-value in output-values
     do (setf (output neuron) (the single-float output-value))))

(defmethod apply-expected-outputs ((net t-net) (expected-output-values list))
  (loop with layer-dlist = (value (tail (layer-dlist net)))
     for neuron-node = (head layer-dlist) then (next neuron-node)
     while neuron-node
     for neuron = (value neuron-node)
     for expected-output-value in expected-output-values
     do (setf (expected-output neuron) (the single-float expected-output-value))))

(defgeneric collect-neurons (thing)
  (:method ((net t-net))
    (loop for layer-node = (head (layer-dlist net)) then (next layer-node)
       while layer-node
       for layer = (value layer-node)
       append (collect-neurons layer)))
  (:method ((layer dlist))
    (loop for neuron-node = (head layer) then (next neuron-node)
       while neuron-node
       collect (value neuron-node))))

(defmethod neuron-count ((net t-net))
  (loop for node = (head (layer-dlist net)) then (next node)
        while node summing (len (value node))))

(defun neuron-by-name (structure name)
  (car (remove-if-not (lambda (n) (equal name (name n)))
                      (collect-neurons structure))))

(defmethod collect-inputs ((net t-net))
  (loop with input-layer-node = (head (layer-dlist net))
     for input-layer = (value input-layer-node)
     for neuron-node = (head input-layer) then (next neuron-node)
     while neuron-node
     for neuron = (value neuron-node)
     collect (input neuron)))

(defmethod collect-all-inputs ((net t-net))
  (loop for layer-node = (head (layer-dlist net)) then (next layer-node)
     while layer-node append
       (loop for neuron-node = (head (value layer-node)) then (next neuron-node)
          while neuron-node
          collect (input (value neuron-node)))))

(defmethod collect-outputs ((net t-net))
  (loop with output-layer-node = (tail (layer-dlist net))
     for output-layer = (value output-layer-node)
     for neuron-node = (head output-layer) then (next neuron-node)
     while neuron-node collect (output (value neuron-node))))

(defmethod collect-output-errors ((net t-net))
  (loop with output-layer-node = (tail (layer-dlist net))
     with output-layer = (value output-layer-node)
     for neuron-node = (head output-layer) then (next neuron-node)
     while neuron-node collect (err (value neuron-node))))

(defmethod collect-expected-outputs ((net t-net))
  (loop with output-layer-node = (tail (layer-dlist net))
     for output-layer = (value output-layer-node)
     for neuron-node = (head output-layer) then (next neuron-node)
     while neuron-node collect (expected-output (value neuron-node))))

(defgeneric collect-cxs (thing)
  (:method ((net t-net))
    (loop for layer-node = (head (layer-dlist net)) then (next layer-node)
          while layer-node
          appending (collect-cxs (value layer-node))))
  (:method ((layer dlist))
    (loop for neuron-node = (head layer) then (next neuron-node)
          while neuron-node
          appending (collect-cxs (value neuron-node))))
  (:method ((neuron t-neuron))
    (loop for cx-node = (head (cx-dlist neuron)) then (next cx-node)
          while cx-node
          collect (value cx-node))))
  
(defgeneric collect-weights (thing)
  (:method ((net t-net))
    (loop for layer-node = (head (layer-dlist net)) then (next layer-node)
       for layer-index = 1 then (1+ layer-index)
       while layer-node
       for layer = (value layer-node) 
       append (collect-weights layer)))
  (:method ((layer dlist))
    (loop for neuron-node = (head layer) then (next neuron-node)
       for neuron-index = 1 then (1+ neuron-index)
       while neuron-node
       for neuron = (value neuron-node) 
       append (collect-weights neuron)))
  (:method ((neuron t-neuron))
    (loop for cx-node = (head (cx-dlist neuron)) then (next cx-node)
       for cx-index = 1 then (1+ cx-index)
       while cx-node
       for cx = (value cx-node)
       collect (weight cx))))

(defun collect-weights-into-file (thing &optional file)
  (when (and (null file) (equal (type-of thing) 't-net))
    (setf file (weights-file thing)))
  (when (null file)
    (error "For other than a t-net object, you must provide the file"))
  (with-open-file (s-out file :direction :output :if-exists :supersede)
    (loop for weight in (collect-weights thing)
       do (write-line (format nil "~f" weight) s-out)))
  file)

(defgeneric apply-weights (thing weights)
  (:method ((net t-net) (weights list))
    (loop for cx in (collect-cxs net)
       for weight in weights
       do (setf (weight cx) weight
                (delta cx) 0.0))))

(defun apply-weights-from-file (thing filename)
  (with-open-file (s-in filename)
    (loop for cx in (collect-cxs thing)
       for weight = (read s-in)
       do (setf (weight cx) weight))))

(defun reset-weights (net)
  (loop 
    initially (reset-random-state net)
    with global-index = 0 
    and global-count = (length (collect-weights net))
    for layer-node = (head (layer-dlist net)) then (next layer-node)
    while layer-node
    for layer = (value layer-node)
    do (loop 
         with layer-index = 0 
         and layer-count = (length (collect-weights layer))
         for neuron-node = (head layer) then (next neuron-node)
         while neuron-node
         for neuron = (value neuron-node)
         do (loop with neuron-count = (length (collect-weights neuron))
                  for cx-node = (head (cx-dlist neuron)) then (next cx-node)
                  while cx-node
                  for cx = (value cx-node)
                  for neuron-index = 0 then (1+ neuron-index)
                  for weight = (funcall (initial-weight-function net)
                                        :rstate (rstate net)
                                        :global-index global-index
                                        :global-fraction (/ (float global-index)
                                                            (float global-count))
                                        :layer-fraction (/ (float layer-index)
                                                           (float layer-count))
                                        :neuron-fraction (/ (float neuron-index)
                                                            (float neuron-count)))
                      do (with-mutex ((weight-mtx cx))
                           (setf (weight cx) weight)
                           (setf (delta cx) 0.0))
                         (incf global-index)
                         (incf layer-index)))))

(defgeneric collect-cxs (thing)
  (:method ((net t-net))
    (loop for layer-node = (head (layer-dlist net)) then (next layer-node)
       while layer-node
       append (collect-cxs (value layer-node))))
  (:method ((layer dlist))
    (loop for neuron-node = (head layer) then (next neuron-node)
       while neuron-node
       append (collect-cxs (value neuron-node))))
  (:method ((neuron t-neuron))
    (loop for cx-node = (head (cx-dlist neuron)) then (next cx-node)
       while cx-node
       collect (value cx-node))))

(defgeneric render-as-text (thing)
  (:method ((net t-net))
    (loop for layer-node = (head (layer-dlist net)) then (next layer-node)
       for layer-index = 1 then (1+ layer-index)
       while layer-node
       for layer = (value layer-node)
       do 
         (format t "Layer ~d~%" layer-index)
         (render-as-text layer)))
  (:method ((layer dlist))
    (loop for neuron-node = (head layer) then (next neuron-node)
       while neuron-node
       for neuron = (value neuron-node)
       do
         (format t "  Neuron ~a ~(~a~) i=~,4f; o=~,4f; eo=~,4f; e=~,4f; ed=~,4f~%" 
                 (name neuron) 
                 (transfer-key neuron)
                 (input neuron)
                 (output neuron)
                 (expected-output neuron)
                 (err neuron)
                 (err-derivative neuron))
         (render-as-text neuron)))
  (:method ((neuron t-neuron))
    (loop for cx-node = (head (cx-dlist neuron)) then (next cx-node)
       while cx-node
       for cx = (value cx-node)
       do (format t "    ~,4f -> ~a~%" (weight cx) (name (target cx))))))

(defun infer (net inputs &key (thread-count *default-thread-count*))
  "NET is a neural network, an object of type T-NET. INPUTS is a list of
input values, each consisting of a floating-point number. The length
of INPUTS must match the number of neurons in the input layer of
NET. This function performs a feed-forward pass of the inputs through
NET and returns a list of output values, each consisting of a
floating-point number. The number of output values will match the
number of neurons in the output layer of NET.

If there's an active thread pool (*thread-pool* is not empty), this
function will use that thread pool. Otherwise, this function will
create a thread pool with THREAD-COUNT threads, use the new thread
pool, then stop it."
  (let ((own-threads (not *thread-pool*)))
    (when own-threads (start-thread-pool thread-count))
    (apply-inputs net inputs)
    (feedforward net)
    (when own-threads (stop-thread-pool))
    (collect-outputs net)))

(defun frame-error (outputs expected-outputs)
  "OUTPUTS is a list of floating-point numbers, each representing the
output of a neuron. EXPECTED-OUTPUTS is a list of floating-point
numbers, each representing the expected output of a neuron. This
function returns the frame error, which is the sum of the squared
differences between the outputs and the expected outputs."
  (loop 
    for actual-output in outputs
    for expected-output in expected-outputs
    sum (expt (- expected-output actual-output) 2)))

(defun train-frame (net frame &key (thread-count *default-thread-count*))
  "NET is a neural network, an object of type T-NET. FRAME is a list with
2 elements: a list of input values and a list of expected output
values. Each value is a floating-point number. The length of the input
list must match the number of neurons in the input layer of NET, and
the length of the output list must match the number of neurons in the
output layer of NET. This function performs a feed-forward pass of the
inputs through NET, then a back-propagation pass of the error between
the actual outputs and the expected output values, and returns the
network error, which is a single floating-point value representing a
measure of the error of the output layer.

If there's an active thread pool (*thread-pool* is not empty), this
function will use that thread pool. Otherwise, this function will
create a thread pool with THREAD-COUNT threads, use the new thread
pool, then stop it."
  (let ((own-threads (not *thread-pool*))
        (inputs (first frame))
        (expected-outputs (second frame)))
    (when own-threads (start-thread-pool thread-count))
    (let* ((outputs (infer net inputs))
           (frame-error (frame-error outputs expected-outputs)))
      (apply-expected-outputs net expected-outputs)
      (backpropagate net)
      (when own-threads (stop-thread-pool))
      frame-error)))

;; (defun train-bad-frame (net
;;                         frame
;;                         target-error
;;                         &key (max-iterations 1)
;;                              (thread-count *default-thread-count*))
;;   "This function repeatedly trains the neural network on FRAME
;; until the network error for the input reaches TARGET-ERROR or until
;; the training iterations exceed MAX-ITERATIONS.

;; NET, is a neural network, an object of type T-NET. FRAME is a list
;; with 2 elements: a list of input values and a list of expected output
;; values. Each value is a floating-point number. The length of the input
;; list must match the number of neurons in the input layer of NET, and
;; the length of the output list must match the number of neurons in the
;; output layer of NET. TARGET-ERROR is a floating-point number that
;; tells the function to stop training when the network error is less
;; than or equal to this value. MAX-ITERATIONS is an integer that tells
;; the function to stop training when the number of iterations exceeds
;; this value. This function performs a feed-forward pass of the inputs
;; through NET, then a back-propagation pass of the error between the
;; actual outputs and the expected output values, and returns network
;; error for the frame."
;;   (let ((own-threads (not *thread-pool*)))
;;     (loop 
;;       initially (when own-threads (start-thread-pool thread-count))
;;       with inputs = (first frame)
;;       with expected-outputs = (second frame)
;;       for iteration = 0 then (1+ iteration)
;;       for outputs = (infer net inputs)
;;       for frame-error = (frame-error outputs expected-outputs)
;;       while (and (< iteration max-iterations)
;;                  (<= frame-error target-error))
;;       do
;;          (apply-expected-outputs net expected-outputs)
;;          (backpropagate net)
;;       finally (when own-threads (stop-thread-pool))
;;               (return frame-error))))

(defmethod train-bad-frame ((net t-net)
                            (inputs list)
                            (expected-outputs list)
                            (target-error float))
  (let ((own-threads (not *thread-pool*)))
    (when own-threads (start-thread-pool *default-thread-count*))
    (let* ((outputs (infer net inputs))
           (frame-error (loop for actual in outputs
                           for expected in expected-outputs
                           summing (expt (- expected actual) 2))))
      (when (> frame-error target-error)
        (apply-expected-outputs net expected-outputs)
        (backpropagate net))
      (when own-threads (stop-thread-pool))
      frame-error)))

(defun set-training-in-progress (thread)
  (if thread
      (with-mutex (*training-in-progress-mutex*)
        (setf *main-training-thread* thread))
      (progn
        (setf *continue-training* nil)
        (sleep 1)
        (setf *main-training-thread* nil)
        (loop for thread in (list-all-threads)
           when (equal (thread-name thread) "main-training-thread")
           do (terminate-thread thread)))))

(defun get-training-in-progress ()
  (with-mutex (*training-in-progress-mutex*)
    *main-training-thread*))

(defmethod train-frames ((environment environment)
                         (training-frames vector)
                         (when-complete function)
                         &key
                           (epochs 6)
                           (target-error 0.05)
                           (reset-weights t)
                           (report-function #'default-report-function)
                           (report-frequency 10))
  (when (get-training-in-progress)
    (error "Training is already in progress."))
  (with-open-file (stream (log-file (net environment))
                          :direction :output
                          :if-does-not-exist :create
                          :if-exists :append)
    (format stream "~%BEGIN~%"))
  (set-training-in-progress
   (make-thread
    (lambda ()
      (when reset-weights (reset-weights (net environment)))
      (let ((result (train-frames-work environment
                                       training-frames
                                       epochs 
                                       target-error 
                                       report-function 
                                       report-frequency)))
        (funcall when-complete
                 (id environment)
                 (getf result :presentations)
                 (getf result :start-time) 
                 (getf result :network-error))))
    :name "main-training-thread")))

(defmethod train-frames-work ((environment environment)
                              (training-frames vector)
                              (epochs integer)
                              (target-error float)
                              (report-function function)
                              (report-frequency integer))
  (loop
    initially (setf *continue-training* t)
    with net = (net environment)
    and start-time = (get-universal-time)
    and presentation = 0
    and last-presentation = 0
    and sample-size = (length training-frames)
    with last-report-time = start-time
    for epoch from 1 to epochs
    for network-error = (loop
                          for index in (shuffle 
                                        (loop for a from 0 below sample-size
                                              collect a)
                                        (rstate net))
                          for count = 0 then (1+ count)
                          for frame = (aref training-frames index)
                          for (inputs expected-outputs) = frame
                          for outputs = (progn 
                                          (incf presentation)
                                          (infer net inputs))
                          for frame-error = (frame-error 
                                             outputs expected-outputs)
                          for network-error = frame-error 
                            then (+ network-error frame-error)
                          when (> frame-error target-error)
                            do (apply-expected-outputs net expected-outputs)
                               (backpropagate net)
                          when (>= (- (get-universal-time) last-report-time)
                                   report-frequency)
                            do (funcall report-function
                                        environment
                                        epoch
                                        count
                                        presentation
                                        last-presentation
                                        (- (get-universal-time) start-time)
                                        (- (get-universal-time) last-report-time)
                                        (/ network-error count))
                               (setf last-report-time (get-universal-time))
                               (setf last-presentation presentation)
                          finally (return (/ network-error sample-size)))
    while (> network-error target-error)
    when (> (- (get-universal-time) last-report-time) report-frequency)
      do (funcall report-function
                  environment
                  epoch 
                  presentation last-presentation
                  (- (get-universal-time) start-time)
                  (- (get-universal-time) last-report-time)
               network-error)
         (setf last-report-time (get-universal-time))
         (setf last-presentation presentation)
    finally (return (list :presentations presentation
                          :start-time start-time 
                          :network-error network-error))))
            

(defun vector-average (v)
  (loop for a across v summing a into total
     finally (return (/ total (length v)))))

(defun stop-training (id)
  (let ((log-file (log-file (net (getf *environments* id)))))
    (when (get-training-in-progress)
      (stop-thread-pool)
      (let ((message (format nil "END")))
        (with-open-file (stream log-file
                                :direction :output
                                :if-exists :append
                                :if-does-not-exist :create)
          (write-line message stream))
        (format t message))
      (sleep 1)
      (set-training-in-progress nil))))

(defun error-subset-count (size training-frames)
  (let ((l (length training-frames)))
    (case size
      (:large (let ((m (truncate l 5)))
                (if (< m 100)
                    l
                    (if (> m 1000) 1000))))
      (:small (let* ((m (truncate l 100)))
                (if (< m 100)
                    (if (> l 100) 100 l)
                    100))))))

(defun default-report-function (environment iteration count 
                                presentation last-presentation 
                                elapsed-seconds since-last-report 
                                network-error)
  (let ((rate (if (zerop since-last-report)
                  0
                  (/ (- presentation last-presentation) since-last-report)))
        (filename (log-file (net environment))))
    (with-open-file (log-stream filename
                                :direction :output 
                                :if-exists :append 
                                :if-does-not-exist :create)
      (format log-stream "t=~ds; i=~d; v=~d; p=~d; r=~fp/s; e=~d~%" 
              elapsed-seconds iteration count presentation rate
              network-error))))

(defun plotting-report-function (environment iteration count 
                                 presentation last-presentation 
                                 elapsed-seconds since-last-report 
                                 network-error)
  (add-training-error environment
                      elapsed-seconds
                      presentation
                      network-error)
  (when (and (plot-errors environment)
             (> (len (training-error environment)) 1))
    (plot-training-error environment))
  (default-report-function environment iteration count 
                           presentation last-presentation 
                           elapsed-seconds since-last-report 
                           network-error))
         
(defun label->outputs (label label-indexes)
  (loop with index = (gethash label label-indexes)
     for a from 0 below (hash-table-count label-indexes)
     collect (if (= a index) 1.0 0.0)))

(defun outputs->label (environment outputs)
  (elt (index->label environment) (index-of-max outputs)))         

(defun label-indexes->index-labels (label-indexes)
  (loop with index-labels = (make-hash-table)
     for label being the hash-keys in label-indexes using (hash-value index)
     do (setf (gethash index index-labels) label)))

(defgeneric evaluate-inference-1hs (net training-frames)
  (:method ((net t-net) (training-frames list))
    (loop with own-threads = (not *thread-pool*)
       initially (when own-threads (start-thread-pool 7))
       for (inputs expected-outputs) in training-frames
       for index = 0 then (1+ index)
       for expected-winner = (index-of-max expected-outputs)
       for outputs = (infer net inputs)
       for winner = (index-of-max outputs)
       for total = 1 then (1+ total)
       for pass = (= winner expected-winner)
       for correct = (if pass 1 0) then (if pass (1+ correct) correct)
       finally 
         (when own-threads (stop-thread-pool))
         (return (list :percent 
                       (float (* 100 (/ correct total)))
                       :total total 
                       :pass correct 
                       :fail (- total correct)))))
  (:method ((net t-net) (training-frames vector))
    (evaluate-inference-1hs net (map 'list 'identity training-frames))))
  
(defgeneric network-error (net frames)
  (:method ((net t-net) (frames list))
    (loop 
       for (inputs expected-outputs) in frames
       for outputs = (infer net inputs)
       for frame-count = 1 then (1+ frame-count)
       for frame-error = (frame-error outputs expected-outputs)
       summing frame-error into total-error
       finally (return (setf *network-error* (/ total-error frame-count)))))
  (:method ((net t-net) (frames vector))
    (network-error net (map 'list 'identity frames))))

(defgeneric faster-network-error (net frames target-error)
  (:method ((net t-net) (frames vector) (target-error float))
    (loop with l = (length frames)
       for t-count = 100 then (* t-count 10)
       for count = (if (< t-count l) t-count l)
       for frames-subset = (choose-from-vector frames count)
       for ne = (network-error net frames-subset)
       while (and (< ne target-error) (< count l))
       finally (return (values ne count))))
  (:method ((net t-net) (frames list) (target-error float))
    (faster-network-error net (map 'vector 'identity frames) target-error)))

(defmethod refresh-frame-errors ((net t-net) (frames vector) (frame-errors vector))
  (loop for (inputs expected-outputs) across frames
     for outputs = (infer net inputs)
     for index = 0 then (1+ index)
     for frame-error = (frame-error outputs expected-outputs)
     do (setf (aref frame-errors index) frame-error)
     finally (return (vector-average frame-errors))))

(defmethod infer-list ((net t-net) (frames list))
  (loop for frame in frames collect (infer net frame)))

(defun create-layer (count &key add-bias (transfer-key :logistic))
  (when (< count 1) (error "Count must be greater than or equal to 1"))
  (when (not (member transfer-key *transfer-function-keys*))
    (error "Unknown transfer-key ~a. Must be one of (~{~a~^, ~})."
           transfer-key *transfer-function-keys*))
  (loop with layer = (make-instance 'dlist)
     and n = (+ count (if add-bias 1 0))
     for a from 1 to n
     for transfer = (getf *transfer-functions* transfer-key)
     for neuron = (make-instance 
                   't-neuron 
                   :id (bianet-id)
                   :biased (and add-bias (= a n))
                   :transfer-key transfer-key)
     do (push-tail layer neuron)
     finally (return layer)))

(defun create-net (topology &key (id (bianet-id)) log-file)
  (loop with log = (or log-file (join-paths *log-folder*
                                            (format nil "~(~a~).log" id)))
     with net = (make-instance 't-net :id id :log-file log)
     for layer-spec in topology
     for count = (or (getf layer-spec :neurons)
                     (error ":neurons parameter required"))
     for add-bias = (getf layer-spec :add-bias)
     for transfer-key = (getf layer-spec :transfer-key :logistic)
     for layer = (create-layer count 
                               :add-bias add-bias 
                               :transfer-key transfer-key)
     do (push-tail (layer-dlist net) layer)
     finally 
       (name-neurons net)
       (return net)))

(defun create-standard-net (succinct-topology 
                            &key 
                              (transfer-function :relu)
                              (id (bianet-id))
                              (weight-reset-function 
                               (make-sinusoid-weight-fn :min -0.5 :max 0.5))
                              (momentum *default-momentum*)
                              (learning-rate *default-learning-rate*)
                              (cx-mode :full)
                              (cx-params 12))
  (loop
     with connect-function = (case cx-mode
                               (:full (lambda (net)
                                        (connect-fully
                                         net
                                         :learning-rate learning-rate
                                         :momentum momentum)))
                               (:partial (lambda (net)
                                           (connect-mostly
                                            net
                                            :learning-rate learning-rate
                                            :momentum momentum
                                            :skip-modulus cx-params)))
                               (otherwise (error "Unknown cx-mode ~(~a~)." cx-mode)))
     with net = (make-instance 't-net :id id
                               :initial-weight-function weight-reset-function
                               :connect-function connect-function)
     with last-layer = (1- (length succinct-topology))
     for neuron-count in succinct-topology
     for layer-index = 0 then (1+ layer-index)
     for in-input-layer = (zerop layer-index)
     for in-output-layer = (= layer-index last-layer)
     for in-hidden-layer = (and (not in-input-layer) (not in-output-layer))
     for transfer-key = (if in-output-layer :logistic transfer-function)
     for layer = (create-layer neuron-count 
                               :add-bias in-hidden-layer 
                               :transfer-key transfer-key)
     do (push-tail (layer-dlist net) layer)
     finally
       (name-neurons net)
       (funcall (connect-function net) net)
       (reset-weights net)
       (create-gates net)
       (setf (log-file net) (default-log-file-name net))
       (setf (weights-file net) (default-weights-file-name net))
       (return net)))
                               
(defmethod create-gates ((net t-net))
  (setf *gates* 
        (map 'vector 'identity
             (loop 
                for layer-node = (head (layer-dlist net)) then (next layer-node)
                while layer-node
                for layer-index = 0 then (1+ layer-index)
                for gate-name = (format nil "~(~a-~d~)" (id net) layer-index)
                collecting (make-gate :name gate-name)))))

(defmethod name-neurons ((net t-net))
  (loop with global-index = 0
     for layer-node = (head (layer-dlist net)) then (next layer-node)
     for layer-index = 1 then (1+ layer-index)
     while layer-node
     for layer = (value layer-node)
     do (loop for neuron-node = (head layer) then (next neuron-node)
           for neuron-index = 1 then (1+ neuron-index)
           while neuron-node
           for neuron = (value neuron-node)
           do (incf global-index)
             (setf (name neuron) 
                   (format nil "~a-~a~a" 
                           layer-index
                           neuron-index
                           (if (biased neuron) "b" ""))))))

(defun connect-fully (net &key 
                            (learning-rate *default-learning-rate*)
                            (momentum *default-momentum*))
  (loop for layer-node = (head (layer-dlist net)) then (next layer-node)
     while (next layer-node)
     for layer = (value layer-node)
     for next-layer = (value (next layer-node))
     do (loop for source-node = (head layer) then (next source-node)
           while source-node do 
             (loop for target-node = (head next-layer) then (next target-node)
                while target-node
                for source = (value source-node)
                for target = (value target-node)
                when (not (biased target))
                do
                  (push-tail 
                   (cx-dlist source)
                   (make-instance 
                    't-cx 
                    :source source 
                    :target target
                    :learning-rate learning-rate
                    :momentum momentum))))))

(defun connect-mostly (net &key 
                             (learning-rate *default-learning-rate*)
                             (momentum *default-momentum*)
                             (skip-modulus 13))
  (loop with index = 0
     for layer-node = (head (layer-dlist net)) then (next layer-node)
     while (next layer-node)
     for layer = (value layer-node)
     for next-layer = (value (next layer-node))
     do (loop for source-node = (head layer) then (next source-node)
           while source-node do 
             (loop for target-node = (head next-layer) then (next target-node)
                while target-node
                for source = (value source-node)
                for target = (value target-node)
                when (and (not (biased target))
                          (not (zerop (mod (incf index) skip-modulus))))
                do 
                  (push-tail 
                   (cx-dlist source)
                   (make-instance 
                    't-cx 
                    :source source 
                    :target target
                    :learning-rate learning-rate
                    :momentum momentum))))))

(defun add-connected-neuron (net hidden-layer-node
                             &key
                               (learning-rate *default-learning-rate*)
                               (momentum *default-momentum*))
  (let ((neuron (make-instance 't-neuron 
                               :id (bianet-id)
                               :biased nil
                               :transfer-key :relu))
        (hidden-layer (value hidden-layer-node))
        (prev-layer (value (prev hidden-layer-node)))
        (next-layer (value (next hidden-layer-node))))
    ;; Add neuron to hidden layer
    (push-tail hidden-layer neuron) 
    ;; Add incoming connections from previous layer
    (add-incoming-cxs net neuron prev-layer
                      :learning-rate learning-rate
                      :momentum momentum)
    ;; Add outgoing connections to next layer
    (add-outgoing-cxs net neuron next-layer
                      :learning-rate learning-rate
                      :momentum momentum))
  (name-neurons net))

(defun add-incoming-cxs (net neuron source-layer
                         &key
                           (learning-rate *default-learning-rate*)
                           (momentum *default-momentum*))
  (loop for source-node = (head source-layer) then (next source-node)
     while source-node
     for source = (value source-node)
     for cx = (make-instance 't-cx
                             :source source
                             :target neuron
                             :learning-rate learning-rate
                             :momentum momentum
                             :weight (random 0.1 (rstate net)))
     do (push-tail (cx-dlist source) cx)))

(defun add-outgoing-cxs (net neuron target-layer
                         &key
                           (learning-rate *default-learning-rate*)
                           (momentum *default-momentum*))
    (loop with cxs = (cx-dlist neuron)
       for target-node = (head target-layer) then (next target-node)
       while target-node
       for target = (value target-node)
       when (not (biased target))
       do (push-tail cxs (make-instance 't-cx
                                        :source neuron
                                        :target target
                                        :learning-rate learning-rate
                                        :momentum momentum
                                        :weight (random 0.1 (rstate net))))))

(defmethod circle-data-1hs ((net t-net) (count integer))
  (loop with true = 0.0 and false = 0.0 and state = (rstate net)
     for a from 1 to count
     for x = (random 1.0 state)
     for y = (random 1.0 state)
     for r = (sqrt (+ (* x x) (* y y)))
     if (< r 0.707) do (setf true 1.0 false 0.0)
     else do (setf true 0.0 false 1.0)
     collect (list (list x y) (list false true))))

(defun shell-execute (program &optional parameters (input-pipe-data ""))
  "Run PROGRAM and return the output of the program as a string.  You can pass an atom or a list for PARAMETERS (the command-line options for the program). You can also pipe data to the program by passing the INPUT-PIPE-DATA parameter with a string containing the data you want to pipe.  The INPUT-PIPE-DATA parameter defaults to the empty string."
  (let ((parameters (cond ((null parameters) nil)
                          ((atom parameters) (list parameters))
                          (t parameters))))
    (with-output-to-string (output-stream)
      (with-output-to-string (error-stream)
        (with-input-from-string (input-stream input-pipe-data)
          (sb-ext:run-program program parameters
                              :search t
                              :output output-stream
                              :error error-stream
                              :input input-stream))))))

(defun file-line-count (filename)
  (let ((result (shell-execute "wc" (list "-l" filename))))
    (if (zerop (length result))
        nil
        (read-from-string result))))

(defun type-1-csv-line->label-and-inputs (csv-line)
  (loop with word = nil
        for c across csv-line
        if (char= c #\,) collect (reverse word) into words and do (setf word nil)
          else do (push c word)
        finally
           (return 
             (loop for chars in (reverse (cons (reverse word) (reverse words)))
                   for first = t then nil
                   for string = (map 'string 'identity chars)
                   collect (if first string (read-from-string string))))))

(defun ensure-data-paths-exist (project-directory train-path test-path)
  (case (path-type project-directory)
    (:not-found (error "project-directory not found: ~a" project-directory))
    (:file (error "project-directory is a file: ~a" project-directory)))
  (let ((path (join-paths project-directory train-path)))
    (when (eql (path-type path) :not-found)
      (error "train-path not found: ~a" path)))
  (when test-path
    (let ((path (join-paths project-directory test-path)))
      (when (eql (path-type path) :not-found)
        (error "test-path not found: ~a" path)))))

(defun update-training-set (id)
  (let* ((env (environment-by-id id))
         (file (training-file env))
         (label-counts (type-1-file->label-counts file))
         (label->index (label-counts->label-indexes label-counts))
         (label->expected-outputs (label-outputs-hash label->index)))
    (setf (training-set env) (file->training-set file label->expected-outputs))
    :done))

(defun file->training-set (file label->expected-outputs)
  (map 'vector 'identity
       (normalize-set (type-1-file->set file label->expected-outputs))))

(defun file->test-set (file label->expected-outputs)
  (normalize-set (type-1-file->set file label->expected-outputs)))

(defun png-file->frame (id label file)
  (let* ((environment (environment-by-id id))
         (inputs (normalize-list (read-png file) :min 0 :max 255))
         (outputs (gethash label (label->expected-outputs environment))))
    (list inputs outputs)))

(defun png-file->pixels (file &key 
                                (target-width 28) 
                                (target-height 28))
  "Read the given PNG file and turns its data into rows of floating
point values that represent the intensity of each pixel, after the
image has been converted to black and white.  The intensity is given
by a floating-point value in the range 1 to 1, where 0 is black and 1
is white.  The length of the rows corresponds to the width of the
image represented by the resulting pixels.  The number of rows represents 
the hight of image represented by the resulting pixels.  That width 
and height is given by the target-width and target-height parameters,
and this function adjusts the image retreived from the file to fit into
the given width and height.
"
  (loop with image-data = (png-read:image-data
                           (png-read:read-png-file file))
        and target-array = (make-array
                            (list target-width target-height))
        with dimensions = (array-dimensions image-data)
        with x-max = (elt dimensions 0)
        and y-max = (elt dimensions 1)
        and c-max = (if (> (length dimensions) 2) (elt dimensions 2) 0)
        with x-delta = (/ (float x-max) (float target-width))
        and y-delta = (/ (float y-max) (float target-height))
        and max-intensity = (float (if (zerop c-max) 255 (* c-max 255)))
        initially (format t "x-max=~d; y-max=~d; c-max=~d; x-delta=~d; y-delta=~d"
                          x-max y-max c-max x-delta y-delta)

        for y-source from 0.0 below y-max by y-delta
        for y-source-int = (truncate y-source)
        for y-target = 0 then (min (1+ y-target) (1- target-height))
        do
           (loop for x-source from 0.0 below x-max by x-delta
                 for x-source-int = (truncate x-source)
                 for x-target = 0 then (min (1+ x-target) (1- target-width))
                 for intensity = (/ (float
                                     (if (zerop c-max)
                                         (aref image-data x-source-int y-source-int)
                                         (loop for c from 0 below c-max 
                                               summing (aref image-data 
                                                             x-source-int
                                                             y-source-int
                                                             c))))
                                    max-intensity)
                 do (setf (aref target-array 
                                (truncate x-target)
                                (truncate y-target))
                             (- 1.0 intensity)))
        finally (return (loop for y from 0 below target-height
                              collect (loop for x from 0 below target-width
                                            collect (aref target-array x y))))))

(defun drop-environment (id)
  (remf *environments* id)
  (when (and *environment* (equal (id *environment*) id))
    (format t "~a :~(~a~) ~a~%"
            "Current environment cleared because"
            id "points to current environment.")
    (setf *environment* nil)
    (setf *net* nil)
    (setf *training-set* nil)
    (setf *test-set* nil))
  (list-environments))

(defun list-environments ()
  (loop for id in *environments* by #'cddr
     for environment in (cdr *environments*) by #'cddr
     collect (list :id id
                   :topology (simple-topology (net environment))
                   :training-set (length (training-set environment))
                   :test-set (length (test-set environment)))))

(defun train (environment-id &key
                   (epochs 100)
                   (target-error 0.05)
                   (reset-weights t)
                   (thread-count *default-thread-count*)
                   (report-function #'plotting-report-function)
                   (report-frequency 10)
                   plot-errors
                   weights-file)
  (let ((environment (getf *environments* environment-id)))
    (when (not environment) (error "No such environment ~(~a~)" environment-id))
    (unless (directory-exists-p (path-only (log-file *net*)))
      (ensure-directories-exist (path-only (log-file *net*))))
    (when (or reset-weights (not weights-file))
      (clear (training-error environment)))
    (when weights-file
      (apply-weights-from-file (net environment) weights-file)
      (setf reset-weights nil))
    (setf (plot-errors environment) plot-errors)
    (when (plot-errors environment)
      (title (format nil "~(~a~) ~{~a~^-~} Training Error"
                     environment-id (simple-topology (net environment))))
      (axis (list t t 0 1.0)))
    (ensure-directories-exist *log-folder*)
    (start-thread-pool thread-count)
    (train-frames environment
                  (training-set environment)
                  #'training-complete
                  :epochs epochs
                  :target-error target-error
                  :reset-weights reset-weights
                  :report-frequency report-frequency
                  :report-function report-function))
  :training)

(defun set-current-environment (id)
  (unless (setf *environment* (environment-by-id id))
    (error "No environment for key ~(~a~)." id))
  (setf *net* (net *environment*)
        *training-set* (training-set *environment*)
        *test-set* (test-set *environment*))
  *environment*)

(defun training-complete (id presentation start-time network-error)
  (stop-thread-pool)
  (let* ((environment (getf *environments* id))
         (fitness (evaluate-inference-1hs (net environment)
                                          (test-set environment)))
         (elapsed-seconds (- (get-universal-time) start-time)))
    (add-training-error environment elapsed-seconds presentation network-error)
    (setf (fitness environment) fitness)
    (when (plot-errors environment)
      (plot-training-error environment))
    (with-open-file (log-stream (log-file (net environment))
                                :direction :output
                                :if-exists :append
                                :if-does-not-exist :create)
      (format log-stream "Result: t=~ds; e=~f; pass=~$% (~d/~d)~%END~%"
              (- (get-universal-time) start-time)
              network-error
              (getf fitness :percent)
              (getf fitness :pass)
              (getf fitness :total)))
    (set-training-in-progress nil)
    (collect-weights-into-file (net environment))))

(defun wait-for-training-completion (id)
  "Waits for training to complete, returns (list fitness% elapsed-seconds presentation network-error)"
  (let ((environment (getf *environments* id)))
    (loop while (get-training-in-progress) do (sleep 1)
       finally (return (cons (getf (fitness environment) :percent)
                             (value (tail (training-error environment))))))))
  
(defun choose-from-vector (vector n)
  (loop with h = (make-hash-table :test 'equal)
     and l = (length vector)
     for a from 1 to n
     for b = (loop for c = (random l)
                while (gethash c h)
                finally (setf (gethash c h) t)
                  (return c))
     collect (elt vector b)))

(defun net-by-id (id)
  (net (getf *environments* id)))

(defun environment-by-id (id)
  (getf *environments* id))

(defun inputs->png (inputs filename)
  (loop with png = (make-instance 'png
                                  :color-type :grayscale
                                  :width 28
                                  :height 28)
     with image = (data-array png)
     for input in inputs
     for x = 0 then (mod (1+ x) 28)
     for y = 0 then (if (zerop x) (1+ y) y)
     do (setf (aref image y x 0) (- 255 (truncate (* input 255))))
     finally (zpng:write-png png filename)))

(defun add-to-training-set (id label inputs
                            &key (update-training-set t)
                              (count 1))
  (let ((file (training-file (environment-by-id id))))
    (with-open-file (out file :direction :output :if-exists :append)
      (loop for a from 1 to count do
           (format out "~a,~{~a~^,~}~%" label inputs)))
    (when update-training-set (update-training-set id))))

(defun training-set->pngs (id path &key label count)
  (loop with environment = (environment-by-id id)
     for (frame-label inputs) in 
       (loop with frame-count = 0
          for frame across (training-set environment)
          for frame-label = (outputs->label environment (second frame))
          for frame-inputs = (car frame)
          for is-match = (or (null label) (equal frame-label label))
          when is-match collect (list frame-label frame-inputs) into input-lists
          and do (incf frame-count)
          when (and count (>= frame-count count)) do (return input-lists)
          finally (return input-lists))
     for index = 1 then (1+ index)
     for dir = (join-paths path frame-label)
     for filename = (join-paths dir (format nil "~3,'0d.png" index))
     do (ensure-directories-exist (concatenate 'string dir "/"))
       (inputs->png inputs filename)))

(defun classify-pngs (environment path prefix)
  (declare (ignore environment))
  (loop with dir-spec = (format nil "~a-*.png" (join-paths path prefix))
     for file in (directory dir-spec)
     for normalized-file-data = (normalize-list (read-png file) :min 0 :max 255)
     for outputs = (infer *net* normalized-file-data)
     for label = (outputs->label *environment* outputs)
     collect (list (file-namestring file) label)))

(defun train-on-png (id label file count)
  (loop with environment = (environment-by-id id)
     with net = (net environment)
     with frame = (png-file->frame :digits label file)
     with inputs = (car frame)
     with expected-outputs = (second frame)
     for a from 1 to count do (train-frame net frame)
     finally (return (outputs->label environment (infer net inputs)))))

(defun list->key-index (string-list)
  (loop with key-index = (make-hash-table :test 'equal)
     for key in (sort string-list #'string<)
     for index = 0 then (1+ index)
     do (setf (gethash key key-index) index)
     finally (return key-index)))

(defun hash-keys (hash-table)
  (loop for key being the hash-keys in hash-table collect key))

(defun key-index->vector (hash-table)
  (let ((keys (sort (hash-keys hash-table)
                    (lambda (a b) (< (gethash a hash-table)
                                     (gethash b hash-table))))))
    (map 'vector 'identity keys)))
       
(defun pngs->frames-for-label (label-folder expected-outputs &key as-vector)
  (loop 
    with dir-spec = (format nil "~a/*.png" label-folder)
    for file in (directory dir-spec)
    for inputs = (normalize-list (read-png file) :min 0 :max 255)
    collect (list inputs expected-outputs) into frames
    finally (return (if as-vector
                        (map 'vector 'identity frames)
                        frames))))


;; (defun evaluate-topologies (&key
;;                               (id :zero-or-one)
;;                               (inputs 784)
;;                               (outputs 2)
;;                               (hidden-start 16)
;;                               (hidden-stop 32)
;;                               (hidden-step 16)
;;                               (training-file "mnist-0-1-train.csv")
;;                               (test-file "mnist-0-1-test.csv")
;;                               (init-weights-function (make-random-weight-fn))
;;                               (report-frequency 1))
;;   (loop with row-format = "|~{ ~a | ~}"
;;      for hidden from hidden-start to hidden-stop by hidden-step
;;      for environment = (create-environment
;;                         id (list inputs hidden outputs)
;;                         :training-file training-file
;;                         :test-file test-file
;;                         :weight-reset-function init-weights-function)
;;      for training = (progn (train id :report-frequency report-frequency)
;;                            (format t "~a~%" (log-file *net*)))
;;      for (fitness elapsed presentations network-error) =
;;        (wait-for-training-completion id)
;;      do (format t "fit=~,2f%; hidden=~d; secs=~d; presented=~d; error=~d~%"
;;                 fitness hidden elapsed presentations network-error)
;;      collect (format nil row-format
;;                      (list fitness hidden elapsed presentations network-error))
;;      into lines
;;      finally (push
;;               (format nil row-format
;;                       '("fitness" "hidden" "elapsed" "presentations"
;;                         "network-error"))
;;               lines)
;;        (return (format nil "~{~a~%~}" lines))))

;; (defun evaluate-convergence-variance (&key
;;                                         (id :zero-or-none)
;;                                         (inputs 784)
;;                                         (outputs 2)
;;                                         (hidden-units 16)
;;                                         (iterations 5)
;;                                         (training-file "mnist-0-1-train.csv")
;;                                         (test-file "mnist-0-1-test.csv")
;;                                         (init-weights-function (make-random-weight-fn))
;;                                         (report-frequency 1))
;;   (loop with row-format = "|~{ ~a | ~}"
;;      for iteration from 1 to iterations
;;      for environment = (create-environment
;;                         id (list inputs hidden-units outputs)
;;                         :training-file training-file
;;                         :test-file test-file
;;                         :weight-reset-function init-weights-function)
;;      for training = (progn (train id :report-frequency report-frequency)
;;                            (format t "~a~%" (log-file *net*)))
;;      for (fitness elapsed presentations network-error) =
;;        (wait-for-training-completion id)
;;      do (format t "fit=~,2f%; hidden=~d; secs=~d; presented=~d; error=~d~%"
;;                 fitness hidden-units elapsed presentations network-error)
;;      collect (format nil row-format
;;                      (list fitness hidden-units elapsed presentations network-error))
;;      into lines
;;      finally (push
;;               (format nil row-format
;;                       '("fitness" "hidden" "elapsed" "presentations"
;;                         "network-error"))
;;               lines)
;;        (return (format nil "~{~a~%~}" lines))))

;; (defun evaluate-topologies (&key
;;                               (id :zero-or-one)
;;                               (inputs 784)
;;                               (outputs 2)
;;                               (hidden-start 16)
;;                               (hidden-stop 32)
;;                               (hidden-step 16)
;;                               (training-file "mnist-0-1-train.csv")
;;                               (test-file "mnist-0-1-test.csv")
;;                               (init-weights-function (make-random-weight-fn))
;;                               (report-frequency 1))
;;   (loop with row-format = "|~{ ~a | ~}"
;;      for hidden from hidden-start to hidden-stop by hidden-step
;;      for environment = (create-environment
;;                         id 
;;                         :topology (list inputs hidden outputs)
;;                         :training-file training-file
;;                         :test-file test-file
;;                         :weight-reset-function init-weights-function)
;;      for training = (progn (train id :report-frequency report-frequency)
;;                            (format t "~a~%" (log-file *net*)))
;;      for (fitness elapsed presentations network-error) =
;;        (wait-for-training-completion id)
;;      do (format t "fit=~,2f%; hidden=~d; secs=~d; presented=~d; error=~d~%"
;;                 fitness hidden elapsed presentations network-error)
;;      collect (format nil row-format
;;                      (list fitness hidden elapsed presentations network-error))
;;      into lines
;;      finally (push
;;               (format nil row-format
;;                       '("fitness" "hidden" "elapsed" "presentations"
;;                         "network-error"))
;;               lines)
;;        (return (format nil "~{~a~%~}" lines))))

;; (defun evaluate-convergence-variance (&key
;;                                         (id :zero-or-none)
;;                                         (inputs 784)
;;                                         (outputs 2)
;;                                         (hidden-units 16)
;;                                         (iterations 5)
;;                                         (training-file "mnist-0-1-train.csv")
;;                                         (test-file "mnist-0-1-test.csv")
;;                                         (init-weights-function (make-random-weight-fn))
;;                                         (report-frequency 1))
;;   (loop with row-format = "|~{ ~a | ~}"
;;      for iteration from 1 to iterations
;;      for environment = (create-environment
;;                         id 
;;                         :topology (list inputs hidden-units outputs)
;;                         :training-file training-file
;;                         :test-file test-file
;;                         :weight-reset-function init-weights-function)
;;      for training = (progn (train id :report-frequency report-frequency)
;;                            (format t "~a~%" (log-file *net*)))
;;      for (fitness elapsed presentations network-error) =
;;        (wait-for-training-completion id)
;;      do (format t "fit=~,2f%; hidden=~d; secs=~d; presented=~d; error=~d~%"
;;                 fitness hidden-units elapsed presentations network-error)
;;      collect (format nil row-format
;;                      (list fitness hidden-units elapsed presentations network-error))
;;      into lines
;;      finally (push
;;               (format nil row-format
;;                       '("fitness" "hidden" "elapsed" "presentations"
;;                         "network-error"))
;;               lines)
;;        (return (format nil "~{~a~%~}" lines))))

                                        

