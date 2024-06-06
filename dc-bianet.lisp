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
    (cond ((< count 3) count)
          ((= count 3) 2)
          (t (1+ (truncate count 2))))))
(defparameter *next-id* 0)
(defparameter *id-mutex* (make-mutex :name "id"))
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
  "Returns an incrementing integer that can serve as a unique ID."
  (with-mutex (*id-mutex*)
    (incf *next-id*)))

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
  ((id :reader id :type integer :initform (bianet-id))
   (source :reader source :initarg :source :type t-neuron
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
   (update-count :accessor update-count :type integer :initform 0)
   (weight-mtx :reader weight-mtx :initform (make-mutex))))

(defclass t-neuron ()
  ((id :reader id :type integer :initform (bianet-id))
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
  (when (zerop (length (name neuron)))
    (setf (name neuron) (format nil "~d" (id neuron))))
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
  ((id :reader id :type integer :initform (bianet-id))
   (name :accessor name :type string :initarg :name :initform "")
   (layer-dlist :accessor layer-dlist :type dlist :initform (make-instance 'dlist))
   (log-file :accessor log-file :initarg :log-file :initform nil)
   (weights-file :accessor weights-file :initarg :weights-file :initform nil)
   (rstate :accessor rstate :initform (make-random-state (reference-random-state)))
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

(defgeneric delete-neuron (net neuron)
  (:method ((net t-net) (neuron t-neuron))
    (let ((cx-nodes (loop
                      for cx-node = (head (cx-dlist neuron))
                        then (next cx-node)
                      while cx-node
                      when (= (id (target (value cx-node))) (id neuron))
                        collect cx-node)))
      (loop
        for cx-node in cx-nodes
        do (delete-node (cx-dlist neuron) cx-node))
      (loop
        for layer-node = (head (layer-dlist net)) then (next layer-node)
        while layer-node
        for layer = (value layer-node)
        for layer-index = 1 then (1+ layer-index)
        for neuron-node = (loop
                            for node = (head layer) then (next node)
                            while node
                            when (= (id (value node)) (id neuron))
                              do (return node))
        when neuron-node
          do (delete-node layer neuron-node)
             (return (let* ((neuron (value neuron-node))
                            (name (name neuron))
                            (id (id neuron)))
                       (list :name name
                             :id id
                             :layer layer-index
                             :neuron neuron))))))
  (:method ((net t-net) (neuron t))
    (when neuron
      (error "Invalid type for neuron parameter"))))

(defgeneric add-neuron (net layer neuron)
  (:documentation
   "Adds NEURON to LAYER. If the specified layer exists and the function
adds NEURON, then the function returns the layer dlist-node to which
the neuron was added. If the function fails to add NEURON, it returns
NIL.

If the caller wants to add connections to NEURON after this call, then
the value that this function returns upon success, the dlist-node of
the layer where the neuron was added, can be helpful in identifying
layers with neurons that should be connected to NEURON.")
  (:method ((net t-net) (layer-index integer) (neuron t-neuron))
    (loop for layer-node = (head (layer-dlist net))
            then (next layer-node)
          while layer-node
          for layer = (value layer-node)
          for index = 0 then (1+ index)
          when (equal index layer-index)
            do (push-tail layer neuron)
               (return layer-node))))

(defgeneric add-neuron-with-connections (net 
                                         layer 
                                         neuron 
                                         &key 
                                           learning-rate
                                           momentum)
  (:documentation
   "Adds NEURON to LAYER and then connects the neuron to other neurons
 in the network. This is acchieved by calling the functions ADD-NEURON
 and CONNECT-NEURON.")
   (:method ((net t-net) (layer-index integer) (neuron t-neuron) 
             &key 
               (learning-rate *default-learning-rate*)
               (momentum *default-momentum*))
     (let ((layer-node (add-neuron net layer-index neuron)))
       (connect-neuron neuron 
                       layer-node 
                       :learning-rate learning-rate
                       :momentum momentum))))

(defgeneric add-generic-neuron (net layer &key learning-rate momentum)
  (:documentation
   "Adds a generic neuron to LAYER and then connects the neuron to other
neurons in the network. This function works only for hidden layers.")
   (:method ((net t-net) (layer-index integer)
             &key
               (learning-rate *default-learning-rate*)
               (momentum *default-momentum*))
     (let* ((old-topology (simple-topology net))
            (old-cxs-count (length (collect-cxs net)))
            (neuron (make-instance 't-neuron))
            (layer-node (add-neuron net layer-index neuron)))
       (when (or (null (prev layer-node))
                 (null (next layer-node)))
         (error "Hidden layer required."))
       (list :neuron-cxs (connect-neuron neuron
                                         layer-node
                                         :learning-rate learning-rate
                                         :momentum momentum)
             :old-topology old-topology
             :new-topology (simple-topology *net*)
             :old-cxs-count old-cxs-count
             :new-cxs-count (length (collect-cxs net))))))

(defgeneric delete-cx (net source target)
  (:method ((net t-net) (source-name string) (target-name string))
    (loop with neuron = (neuron-by-name net source-name)
          with cx-dlist = (cx-dlist neuron)
          for cx-node = (head cx-dlist) then (next cx-node)
          while cx-node
          for cx = (value cx-node)
          when (equal (name (target cx)) target-name)
            do (let ((cx (delete-node cx-dlist cx-node)))
                 (return cx)))))

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
    "Information about the labels required to create the training and
testing data, as well as for inference. See the labels module for more
information about labels.")
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
   (training-progress
    :accessor training-progress :type dlist :initform (make-instance 'dlist)
    :documentation
    "Contains the accuracy values, as training progresses, so that they can
be graphed during and after training. Each element of this dlist
consists of the following values: elapsed-time, presentation-number,
and accuracy.

Elapsed time is the number of seconds since training began.

Presentation number is the the number of times the neural network has
seen an example from the training set.  If there are 10 examples total
in the training set, and the neural network has seen each 10 times,
then presentation-number will be 100.

Accuracy is a number from 0.0 to 1.0, representing the percentage of
training vectors for which the network currently predicts the correct
output.")
   (training-progress-limit
    :accessor training-progress-limit :type integer 
    :initarg :training-progress-limit
    :initform 10000 :documentation
    "The number of elements to keep in the TRAINING-PROGRESS list. When the
environment reaches this number of elements, the oldest elements fall off
the list.")
   (plot-progress
    :accessor plot-progress :initarg :plot-progress
    :initform nil :documentation
    "A boolean value that, when true, indicates that the environment should
plot progress during training.")
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
         (test-set (create-sample-set test-path-abs
                                      (label-outputs output-labels)
                                      transformation))
         (topology (create-topology hidden-layer-topology training-set))
         (net (create-standard-net
               topology
               (format nil "~(~a~)" id)
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
                                     :plot-progress t)))
    (setf (getf *environments* id) environment)
    (when make-current (set-current-environment id))
    environment))

(defun create-topology (hidden-layer-topology training-set)
  (flatten
   (list (length (car (elt training-set 0)))
         hidden-layer-topology
         (length (second (elt training-set 0))))))

(defun failing-frames (environment-id
                       &key
                         set
                         (thread-count *default-thread-count*))
  (loop
    with environment = (environment-by-id environment-id)
    with frame-set = (or set (test-set environment))
    and label-vector = (label-vector (output-labels environment))
    and net = (net environment)
          initially (start-thread-pool thread-count)
    for (inputs expected-outputs info) across frame-set
    for index = 0 then (1+ index)
    for outputs = (infer net inputs)
    for label = (outputs-label label-vector outputs)
    for expected-label = (outputs-label label-vector expected-outputs)
    unless (string= label expected-label)
      collect (list :info info
                    :index index
                    :label label
                    :expected-label expected-label
                    :outputs outputs
                    :expected-outputs expected-outputs)
        into failed-tests
    finally
       (stop-thread-pool)
       (return failed-tests)))

(defun infer-frame (environment-id
                    frame-index
                    &key
                      set
                      (thread-count *default-thread-count*))
  (let* ((environment (environment-by-id environment-id))
         (net (net environment))
         (set (or set (test-set environment)))
         (frame (aref set frame-index))
         (inputs (car frame))
         (expected-outputs (second frame))
         (info (third frame))
         (label-vector (label-vector (output-labels environment)))
         (expected-label (outputs-label label-vector expected-outputs))
         (outputs (infer net inputs :thread-count thread-count))
         (label (outputs-label label-vector outputs)))
    (list :correct (string= label expected-label)
          :label label
          :expected-label expected-label
          :info info
          :outputs outputs
          :expected-outputs expected-outputs)))

(defmethod add-training-progress ((environment environment)
                                  (elapsed-seconds integer)
                                  (presentation integer)
                                  (accuracy float))
  (push-tail (training-progress environment)
             (list elapsed-seconds presentation accuracy))
  (when (> (len (training-progress environment))
           (training-progress-limit environment))
    (pop-head (training-progress environment))))

(defun min-max (list) 
  (loop for y in list
        maximizing y into max
        minimizing y into min
        finally (return (values min max))))

(defun min-max-avg (list) 
  (loop for y in list
        maximizing y into max
        minimizing y into min
        summing y into sum
        counting y into count
        finally (return (values min max (/ (float sum) count)))))

(defun plot-y (label list)
  (multiple-value-bind (min max)
      (min-max list)
    (format-plot 
     *debug* 
     (format nil "set yrange [~,4f : ~,4f]" min max))
    (plot list label)))

(defun plot-xy (label x-list y-list)
  (multiple-value-bind (y-min y-max)
      (min-max y-list)
    (format-plot
     *debug*
     (format nil "set yrange [~,4f : ~,4f]" y-min y-max))
    (plot x-list y-list label)))

(defmethod plot-training-progress ((environment environment))
  (loop for node = (head (training-progress environment)) then (next node)
        while node
        for (elapsed presentation network-progress) = (value node)
        collect elapsed into elapsed-seconds
        collect network-progress into list
        finally (plot-xy "" elapsed-seconds list)))

(defmethod default-log-file-name ((net t-net))
  (join-paths *log-folder*
              (format nil "~a-~{~a~^-~}.log" (name net) (simple-topology net))))

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
  (with-mutex ((weight-mtx cx))
    (multiple-value-bind (new-weight new-delta)
        (compute-new-weight (weight cx)
                            (delta cx)
                            (err-derivative (target cx))
                            (output (source cx))
                            (learning-rate cx)
                            (momentum cx))
      (setf (delta cx) new-delta
            (weight cx) new-weight
            (update-count cx) (1+ (update-count cx))))))

(defun compute-new-weight (old-weight
                           old-delta
                           target-error
                           source-output
                           learning-rate
                           momentum)
  (declare (type single-float 
                 old-weight
                 old-delta
                 target-error
                 source-output
                 learning-rate
                 momentum))
  (let* ((new-delta (+ (* learning-rate target-error source-output)
                       (* momentum old-delta))))
    (values (+ old-weight new-delta) new-delta)))

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

(defmethod neuron-stats ((net t-net) (layer-index integer))
  (loop for neuron in (collect-neurons (at (layer-dlist net) layer-index))
        for weights = (collect-weights neuron)
        collect (multiple-value-bind (min max avg)
                    (min-max-avg weights)
                  (list :neuron (name neuron) :min min :max max :avg avg))))

(defmethod cx-stats ((net t-net) (layer-index integer))
  (loop 
    for neuron in (collect-neurons (at (layer-dlist net) layer-index))
    appending
    (loop 
      for cx in (collect-cxs neuron)
      for index = 0 then (1+ index)
      collect (list :src (name neuron)
                    :tgt (name (target cx))
                    :cx index
                    :w (weight cx)
                    :d (delta cx)))))
          

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

(defgeneric set-momentum (thing momentum)
  (:documentation 
   "Set the momentum in every connection in THING to the value of MOMENTUM
and return the number of connections that were updated. THING can be
an object of type T-NET, a DLIST representing a specific T-NET layer,
or an object of type T-NEURON.")
  (:method ((net t-net) (momentum float))
    (loop 
      for layer-node = (head (layer-dlist net)) then (next layer-node)
      while layer-node
      for layer = (value layer-node)
      summing (set-momentum layer momentum)))
  (:method ((layer dlist) (momentum float))
    (loop
      for neuron-node = (head layer) then (next neuron-node)
      while neuron-node
      for neuron = (value neuron-node)
      summing (set-momentum neuron momentum)))
  (:method ((neuron t-neuron) (momentum float))
    (loop
      for cx-node = (head (cx-dlist neuron)) then (next cx-node)
      while cx-node
      for cx = (value cx-node)
      do (setf (momentum cx) momentum)
      counting cx)))

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
                           (target-accuracy 0.99)
                           (reset-weights t)
                           (report-function #'default-report-function)
                           (report-frequency 5)
                           (thread-count *default-thread-count*))
  (when (get-training-in-progress)
    (error "Training is already in progress."))
  (with-open-file (stream (log-file (net environment))
                          :direction :output
                          :if-does-not-exist :create
                          :if-exists :append)
    (format stream "~%BEGIN ~a=~,4f; ~a=~a; ~a=~:[nil~;t~]; ~a=~d~%"
            "target-accuracy" target-accuracy
            "simple-topology" (simple-topology (net environment))
            "reset-weights" reset-weights
            "thread-count" thread-count))
  (set-training-in-progress
   (make-thread
    (lambda ()
      (when reset-weights (reset-weights (net environment)))
      (let ((result (train-frames-work environment
                                       training-frames
                                       epochs
                                       target-accuracy
                                       report-function
                                       report-frequency)))
        (funcall when-complete
                 (id environment)
                 (getf result :presentations)
                 (getf result :start-time)
                 (getf result :accuracy))))
    :name "main-training-thread")))

(defmethod train-frames-work ((environment environment)
                              (training-frames vector)
                              (epochs integer)
                              (target-accuracy float)
                              (report-function function)
                              (report-frequency integer))
  (loop
    initially (setf *continue-training* t)
    with net = (net environment)
    and start-time = (get-universal-time)
    and presentation = 0
    and last-presentation = 0
    and sample-size = (length training-frames)
    and labels = (label-vector (output-labels environment))
    with last-report-time = start-time
    for epoch from 1 to epochs
    for (average-error max-error correct)
      = (loop
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
          for expected-label = (outputs-label labels expected-outputs)
          for actual-label = (outputs-label labels outputs)
          for correct = (if (equal expected-label actual-label) 1 0)
            then (+ correct (if (equal expected-label actual-label) 1 0))
          for frame-error = (frame-error
                             outputs expected-outputs)
          for error-sum = frame-error
            then (+ error-sum frame-error)
          for max-error = frame-error
            then (if (> frame-error max-error) frame-error max-error)
          unless (equal expected-label actual-label)
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
                        (/ correct (float count)))
               (setf last-report-time (get-universal-time))
               (setf last-presentation presentation)
          finally (return (list (/ error-sum count) max-error correct)))
    for correct-ratio = (/ correct (float sample-size))
    while (< correct-ratio target-accuracy)
    when (> (- (get-universal-time) last-report-time) report-frequency)
      do (funcall report-function
                  environment
                  epoch
                  presentation 
                  last-presentation
                  (- (get-universal-time) start-time)
                  (- (get-universal-time) last-report-time)
                  correct-ratio)
         (setf last-report-time (get-universal-time))
         (setf last-presentation presentation)
    finally (return (list :presentations presentation
                          :start-time start-time
                          :accuracy correct-ratio))))


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

(defun default-report-function (environment 
                                iteration 
                                count
                                presentation 
                                last-presentation
                                elapsed-seconds 
                                since-last-report
                                accuracy)
  (let ((rate (if (zerop since-last-report)
                  0
                  (/ (- presentation last-presentation) since-last-report)))
        (filename (log-file (net environment))))
    (with-open-file (log-stream filename
                                :direction :output
                                :if-exists :append
                                :if-does-not-exist :create)
      (format log-stream "~7,' :ds ~3,' :di ~8,' :dv ~9,' :dp ~7,2fp/s ~6,3fc~%"
              elapsed-seconds 
              iteration 
              count 
              presentation 
              rate
              accuracy))))

(defun plotting-report-function (environment 
                                 iteration count
                                 presentation 
                                 last-presentation
                                 elapsed-seconds 
                                 since-last-report
                                 accuracy)
  (add-training-progress environment
                         elapsed-seconds
                         presentation
                         accuracy)
  (when (and (plot-progress environment)
             (> (len (training-progress environment)) 1))
    (plot-training-progress environment))
  (default-report-function environment 
                           iteration 
                           count
                           presentation 
                           last-presentation
                           elapsed-seconds 
                           since-last-report
                           accuracy))

(defgeneric evaluate-inference-1hs (net training-frames)
  (:method ((net t-net) (training-frames list))
    (loop with own-threads = (not *thread-pool*)
            initially (when own-threads (start-thread-pool 7))
          with start-time = (get-internal-real-time)
          for (inputs expected-outputs) in training-frames
          for index = 0 then (1+ index)
          for expected-winner = (index-of-max expected-outputs)
          for outputs = (infer net inputs)
          for winner = (index-of-max outputs)
          for total = 1 then (1+ total)
          for pass = (= winner expected-winner)
          for correct = (if pass 1 0) then (if pass (1+ correct) correct)
          finally
             (let ((time-span (/ (- (get-internal-real-time)
                                    start-time)
                                 (float internal-time-units-per-second)))
                   (percent (* 100 (/ (float correct) total)))
                   (fail (- total correct)))
               (when own-threads (stop-thread-pool))
               (return (list :percent percent
                             :total total
                             :pass correct
                             :fail fail
                             :time-seconds time-span)))))
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

(defun create-layer (layer-index count &key add-bias (transfer-key :logistic))
  "Create a new layer DLIST with COUNT neurons.

If ADD-BIAS is true, then the new layer will include an addition
biased neuron, bringing its neuron count to COUNT + 1. 

LAYER-INDEX is the zero-based index of the layer that you want to
create, with 0 representing the input layer.

TRANSFER-KEY is a keyword that designates the transfer function. It
defaults to :logistic, but :relu and :relu-leaky are also currently
available. For a complete list of transfer functions and their
associated keywords, see the definition of the *transfer-functions*
parameter."
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
                   :name (format nil "~d-~d" layer-index (1- a))
                   :biased (and add-bias (= a n))
                   :transfer-key transfer-key)
     do (push-tail layer neuron)
     finally (return layer)))

(defun create-net (topology name &key log-file)
  (loop
     with net = (make-instance 't-net 
                               :name name 
                               :connect-function (lambda (net)
                                                   (connect-fully
                                                    net
                                                    :learning-rate learning-rate
                                                    :momentum momentum))
                               :log-file (or log-file ""))
     for layer-spec in topology
     for layer-index = 0 then (1+ layer-index)
     for count = (or (getf layer-spec :neurons)
                     (error ":neurons parameter required"))
     for add-bias = (getf layer-spec :add-bias)
     for transfer-key = (getf layer-spec :transfer-key :logistic)
     for layer = (create-layer layer-index
                               count
                               :add-bias add-bias
                               :transfer-key transfer-key)
     do (push-tail (layer-dlist net) layer)
     finally
       (name-neurons net)
       (return net)))

(defun create-standard-net (succinct-topology
                            name
                            &key
                              (transfer-function :relu)
                              (weight-reset-function
                               (make-sinusoid-weight-fn :min -0.5 :max 0.5))
                              (momentum *default-momentum*)
                              (learning-rate *default-learning-rate*)
                              (cx-mode :full)
                              (cx-params 12)
                              log-file)
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
     with net = (make-instance 't-net 
                               :name name
                               :initial-weight-function weight-reset-function
                               :connect-function connect-function)
     with last-layer = (1- (length succinct-topology))
     for neuron-count in succinct-topology
     for layer-index = 0 then (1+ layer-index)
     for in-input-layer = (zerop layer-index)
     for in-output-layer = (= layer-index last-layer)
     for in-hidden-layer = (and (not in-input-layer) (not in-output-layer))
     for transfer-key = (if in-output-layer :logistic transfer-function)
     for layer = (create-layer layer-index
                               neuron-count
                               :add-bias in-hidden-layer
                               :transfer-key transfer-key)
     do (push-tail (layer-dlist net) layer)
     finally
       (name-neurons net)
       (funcall (connect-function net) net)
       (reset-weights net)
       (create-gates net)
       (setf (log-file net) (or log-file (default-log-file-name net)))
       (ensure-directories-exist (log-file net))
       (uiop:run-program (format nil "touch ~s" (log-file net)))
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

(defun connect-neuron (neuron 
                       neuron-layer
                       &key 
                         (learning-rate *default-learning-rate*)
                         (momentum *default-momentum*))
  "Creates incoming connections to NEURON, from neurons in the previous
layer, and creates outgoing connections from NEURON to neurons in the
next layer. The previous and next layers are the layers the precede
and follow NEURON-LAYER, which is a DLIST-NODE that represents the
layer that contains NEURON.

If NEURON is biased of if NEURON-LAYER is the first layer (the input
layer), then this function creates no incoming connections. 

This function avoids creating connections to biased neurons in the
next layer. And, the function will create no outgoing connections if
NEURON-LAYER represents the last layer (the output layer).

Keep in mind that if you add neurons to the input or output layers, you
have to retrain the neural network with training and test sets that
have a different shape.

LEARNING-RATE and MOMENTUM can be set to floating-point values with
their respective keywords, but they have defaults and they can be
changed later."
  ;; Incoming connections
  (let ((incoming 0)
        (outgoing 0))
    (unless (or (biased neuron) (null (prev neuron-layer)))
      (loop for neuron-node = (head (value (prev neuron-layer)))
              then (next neuron-node)
            while neuron-node
            for source-neuron = (value neuron-node)
            for cx = (make-instance 't-cx
                                    :momentum momentum
                                    :learning-rate learning-rate
                                    :weight (+ .25 (random 0.5))
                                    :source source-neuron
                                    :target neuron)
            do (push-tail (cx-dlist source-neuron) cx)
               (incf incoming)))
    (unless (null (next neuron-layer))
      (loop for neuron-node = (head (value (next neuron-layer))
              then (next neuron-node)
            while neuron-node
            for target-neuron = (value neuron-node)
            for cx = (make-instance 't-cx
                                    :momentum momentum
                                    :learning-rate learning-rate
                                    :weight (+ .25 (random 0.5))
                                    :source neuron
                                    :target target-neuron)
            do (push-tail (cx-dlist neuron) cx)
               (incf outgoing))))
    (list :incoming incoming :outgoing outgoing)))
    

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
                   (target-accuracy 0.95)
                   (reset-weights t)
                   (thread-count *default-thread-count*)
                   (report-function #'plotting-report-function)
                   (report-frequency 5)
                   plot-progress
                   weights-file)
  (let ((environment (getf *environments* environment-id)))
    (when (not environment) (error "No such environment ~(~a~)" environment-id))
    (unless (directory-exists-p (path-only (log-file *net*)))
      (ensure-directories-exist (path-only (log-file *net*))))
    (when (or reset-weights (not weights-file))
      (clear (training-progress environment)))
    (when weights-file
      (apply-weights-from-file (net environment) weights-file)
      (setf reset-weights nil))
    (setf (plot-progress environment) plot-progress)
    (when (plot-progress environment)
      (title (format nil "~(~a~) ~{~a~^-~} Accuracy"
                     environment-id (simple-topology (net environment))))
      (axis (list t t 0 1.0)))
    (ensure-directories-exist *log-folder*)
    (start-thread-pool thread-count)
    (train-frames environment
                  (training-set environment)
                  #'training-complete
                  :epochs epochs
                  :target-accuracy target-accuracy
                  :reset-weights reset-weights
                  :thread-count thread-count
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

(defun training-complete (id presentation start-time accuracy)
  (stop-thread-pool)
  (let* ((environment (getf *environments* id))
         (elapsed-seconds (- (get-universal-time) start-time)))
    (add-training-progress environment elapsed-seconds presentation accuracy)
    (compute-fitness id)
    (when (plot-progress environment)
      (plot-training-progress environment))
    (with-open-file (log-stream (log-file (net environment))
                                :direction :output
                                :if-exists :append
                                :if-does-not-exist :create)
      (format log-stream "Result: t=~ds; e=~f; pass=~$% (~d/~d)~%END~%"
              (- (get-universal-time) start-time)
              accuracy
              (getf (fitness environment) :percent)
              (getf (fitness environment) :pass)
              (getf (fitness environment) :total)))
    (set-training-in-progress nil)
    (collect-weights-into-file (net environment))))

(defun compute-fitness (environment-id)
  (let* ((environment (environment-by-id environment-id))
         (net (net environment))
         (test-set (test-set environment)))
    (setf (fitness environment) (evaluate-inference-1hs net test-set))))

(defun wait-for-training-completion (id)
  "Waits for training to complete, returns (list fitness% elapsed-seconds presentation network-error)"
  (let ((environment (getf *environments* id)))
    (loop while (get-training-in-progress) do (sleep 1)
       finally (return (cons (getf (fitness environment) :percent)
                             (value (tail (training-progress environment))))))))

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
