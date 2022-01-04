(in-package :dc-bianet)

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

(defparameter *log-folder* "/tmp/bianet-logs")
(defparameter *db-log-file* "db.log")

(defparameter *environments* nil) ;; p-list of id -> environment

;; Current environment
(defparameter *environment* nil)
(defparameter *net* nil)
(defparameter *network-error* 0.0)
(defparameter *training-set* nil)
(defparameter *test-set* nil)

(defun db-log (message)
  (with-open-file (out (join-paths *log-folder* *db-log-file*)
                       :direction :output
                       :if-exists :append
                       :if-does-not-exist :create)
    (write-line (timestamp :string message) out)))

;; Database
(defparameter *db* (funcall #'ds (cons :map (slurp-n-thaw (join-paths *home-folder* "db-conf.lisp")))))
(ds-set *db* :log-function #'db-log)
(ensure-directories-exist *log-folder*)

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
   (learning-rate :accessor learning-rate :initarg :learning-rate 
                  :type single-float :initform 0.02)
   (momentum :accessor momentum :initarg :momentum :type single-float 
             :initform 0.1)
   (delta :accessor delta :initarg :delta :type single-float :initform 0.0)
   (weight-mtx :reader weight-mtx :initform (make-mutex))))

(defclass t-neuron ()
  ((id :accessor id :initarg :id :type keyword :initform (bianet-id))
   (name :accessor name :initarg :name :type string :initform nil)
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
	 (neuron-count :accessor neuron-count :initform 0)
	 (cx-count :accessor cx-count :initform 0)
	 (width :accessor width :initform nil)
	 (layer-count :accessor layer-count :initform nil)
	 (neuron-index :accessor neuron-index :initform (make-hash-table :test 'equal))
	 (max-weight :accessor max-weight :initform 0)
	 (min-weight :accessor min-weight :initform 0)
   (log-file :accessor log-file :initarg :log-file :initform nil)
   (weights-file :accessor weights-file :initarg :weights-file :initform nil)
   (rstate :reader rstate :initform (make-random-state))
   (initial-weight-function 
    :accessor initial-weight-function
    :initarg :initial-weight-function
    :initform (make-progressive-weight-fn :min -0.5 :max 0.5))
   (connect-function
    :accessor connect-function
    :initarg :connect-function
    :initform (error ":connect-function is required"))))

(defclass t-environment ()
  ((id :accessor id :initarg :id :type keyword)
   (net :accessor net :initarg :net :type t-net)
   (training-file :accessor training-file
                  :initarg :training-file
                  :type string)
   (test-file :accessor test-file
              :initarg :test-file
              :type string)
   (training-set :accessor training-set
                 :initarg :training-set
                 :initform (vector)
                 :type vector)
   (test-set :accessor test-set
             :initarg :test-set
             :initform nil
             :type list)
   (label->index :accessor label->index
                 :initarg :label->index
                 :type hashtable)
   (label->expected-outputs :accessor label->expected-outputs
                            :initarg :label->expected-outputs
                            :type hashtable)
   (index->label :accessor index->label
                 :initarg :index->label
                 :type vector)
   (training-error :accessor training-error
                   :type dlist
                   :initform (make-instance 'dlist))
   (training-error-limit :accessor training-error-limit
                         :type integer
                         :initarg :training-error-limit
                         :initform 1000)
   (plot-errors :accessor plot-errors
                :initarg :plot-errors
                :initform nil)
   (fitness :accessor fitness :type list :initform nil)))
                

(defmethod add-training-error ((environment t-environment)
                               (elapsed-seconds integer)
                               (presentation integer)
                               (network-error float))
  (push-tail (training-error environment)
             (list elapsed-seconds presentation network-error))
  (when (> (len (training-error environment))
           (training-error-limit environment))
    (pop-head (training-error environment))))

(defmethod plot-training-error ((environment t-environment))
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
  (loop with global-index = 0 
     and global-count = (length (collect-weights net))
     for layer-node = (head (layer-dlist net)) then (next layer-node)
     while layer-node
     for layer = (value layer-node)
     do (loop with layer-index = 0 
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

(defun reset-neuron-weights (net neuron)
  (loop with neuron-count = (length (collect-weights neuron))
     for cx-node = (head (cx-dlist neuron)) then (next cx-node)
     while cx-node
     for cx = (value cx-node)
     for neuron-index = 0 then (1+ neuron-index)
     for weight = (the single-float 
                       (funcall (initial-weight-function net)
                                :rstate (rstate net)
                                :global-index neuron-index
                                :global-fraction 0.0
                                :layer-fraction 0.0
                                :neuron-fraction (/ (float neuron-index)
                                                    (float neuron-count))))
     do (with-mutex ((weight-mtx cx))
          (setf (weight cx) weight)
          (setf (delta cx) (the single-float 0.0)))))

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

(defmethod infer-frame ((net t-net) (inputs list))
  (let ((own-threads (not *thread-pool*)))
    (when own-threads (start-thread-pool *default-thread-count*))
    (apply-inputs net inputs)
    (feedforward net)
    (when own-threads (stop-thread-pool))
    (collect-outputs net)))

(defmethod train-frame ((net t-net) (inputs list) (expected-outputs list))
  (let ((own-threads (not *thread-pool*)))
    (when own-threads (start-thread-pool *default-thread-count*))
    (let* ((outputs (infer-frame net inputs))
           (frame-error (loop for actual in outputs
                           for expected in expected-outputs
                           summing (expt (- expected actual) 2))))
      (apply-expected-outputs net expected-outputs)
      (backpropagate net)
      (when own-threads (stop-thread-pool))
      frame-error)))

(defmethod train-bad-frame ((net t-net)
                            (inputs list)
                            (expected-outputs list)
                            (target-error float))
  (let ((own-threads (not *thread-pool*)))
    (when own-threads (start-thread-pool *default-thread-count*))
    (let* ((outputs (infer-frame net inputs))
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

(defmethod train-frames ((environment t-environment)
                         (training-frames vector)
                         (when-complete function)
                         &key
                           (epochs 6)
                           (target-error 0.05)
                           (reset-weights t)
                           (report-function #'default-report-function)
                           (report-frequency 10)
                           (skip-refresh t))
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
                                       report-frequency
                                       skip-refresh)))
        (funcall when-complete
                 (id environment)
                 (getf result :presentations)
                 (getf result :start-time) 
                 (getf result :network-error))))
    :name "main-training-thread")))

(defmethod train-frames-work ((environment t-environment)
                              (training-frames vector)
                              (epochs integer)
                              (target-error float)
                              (report-function function)
                              (report-frequency integer)
                              (skip-refresh t))
  (loop initially 
       (setf *continue-training* t)
     with net = (net environment)
     and start-time = (get-universal-time)
     and presentation = 0
     and last-presentation = 0
     and sample-size = (length training-frames)
     with last-report-time = start-time
     with frame-errors = (make-array sample-size 
                                     :element-type 'float 
                                     :initial-element 1.0)
     for indexes = (shuffle (loop for a from 0 below sample-size collect a))
     for epoch from 1 to epochs
     for network-error = (average frame-errors)
     while (and
            (if skip-refresh
                (> network-error target-error)
                (or (> network-error target-error)
                    (> (refresh-frame-errors net training-frames frame-errors)
                       target-error)))
            *continue-training*)
     do (loop 
           for index in indexes
           for (inputs expected-outputs) = (aref training-frames index)
           for count = 1 then (1+ count)
           for elapsed-seconds = (- (get-universal-time) start-time)
           for since-last-report = (- (get-universal-time) last-report-time)
           for frame-error = (aref frame-errors index)
           while *continue-training*
           when (> frame-error target-error)
           do
             (incf presentation)
             (setf (aref frame-errors index)
                   (train-bad-frame net inputs expected-outputs target-error))
           when (>= since-last-report report-frequency)
           do
             (funcall report-function
                      environment
                      epoch count presentation last-presentation
                      elapsed-seconds since-last-report
                      (average frame-errors))
             (setf last-report-time (get-universal-time))
             (setf last-presentation presentation))
     finally (return (list :presentations presentation
                           :start-time start-time 
                           :network-error network-error))))

(defun average (v)
  (loop for a across v summing a into total counting a into count
     finally (return (/ total count))))

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

(defgeneric normalize-set (set)
  (:method ((set list))
    (loop with max-value = (loop for frame in set
                              for inputs = (car frame)
                              maximizing (loop for input in inputs 
                                            maximizing input))
       with min-value = (loop for frame in set
                           for inputs = (car frame)
                           minimizing (loop for input in inputs
                                         minimizing input))
       with range = (float (- max-value min-value))
       for frame in set
       for input-values = (car frame)
       for expected-outputs = (second frame)
       collect 
         (list
          (loop for input-value in input-values 
             for new-value = (/ (- input-value min-value) range)
             collect new-value)
          expected-outputs)))
  (:method ((set array))
    (normalize-set (map 'list 'identity set))))


(defun hash-table-to-plist (hash-table)
  (loop for k being the hash-keys in hash-table using (hash-value v)
       collect (list k v)))

(defun type-1-file->set (filename label->expected-outputs)
  (let* ((line-count (file-line-count filename))
         (set (make-array line-count :element-type 'list :initial-element nil)))
    (with-open-file (file filename)
      (loop for line = (read-line file nil)
         for index = 0 then (1+ index)
         while (and line (> (length line) 1))
         for values = (type-1-csv-line->label-and-inputs line)
         for label = (car values)
         for inputs = (cdr values)
         for expected-outputs = (gethash label label->expected-outputs)
         do (setf (aref set index) (list inputs expected-outputs))))
    set))
         
(defun label-outputs-hash (label-index)
  (loop with label-outputs = (make-hash-table :test 'equal)
     for label being the hash-keys in label-index using (hash-value index)
     do (setf (gethash label label-outputs)
              (label->outputs label label-index))
     finally (return label-outputs)))

(defun label->outputs (label label-indexes)
  (loop with index = (gethash label label-indexes)
     for a from 0 below (hash-table-count label-indexes)
     collect (if (= a index) 1.0 0.0)))

(defun outputs->label (environment outputs)
  (elt (index->label environment) (index-of-max outputs)))
         
(defun label-counts->label-indexes (label-counts)
  (loop with label-indexes = (make-hash-table :test 'equal)
     for label in (sort (loop for label being the hash-keys in label-counts
                           collect label)
                        #'string<)
     for index = 0 then (1+ index)
     do (setf (gethash label label-indexes) index)
     finally (return label-indexes)))

(defun labels->label-indexes (list)
  (loop with label-indexes = (make-hash-table :test 'equal)
     for label in (sort list #'string<)
     for index = 0 then (1+ index)
     do (setf (gethash label label-indexes) index)
     finally (return label-indexes)))

(defun label-indexes->index-labels (label-indexes)
  (loop with index-labels = (make-hash-table)
     for label being the hash-keys in label-indexes using (hash-value index)
     do (setf (gethash index index-labels) label)))

(defun type-1-file->label-counts (filename)
  (let ((label-counts (make-hash-table :test 'equal)))
    (with-open-file (file filename)
      (loop for line = (read-line file nil)
         while line
         for label = (subseq line 0 (search "," line))
         do (incf (gethash label label-counts 0))))
    label-counts))

(defgeneric evaluate-inference-1hs (net training-frames)
  (:method ((net t-net) (training-frames list))
    (loop with own-threads = (not *thread-pool*)
       initially (when own-threads (start-thread-pool 7))
       for (inputs expected-outputs) in training-frames
       for index = 0 then (1+ index)
       for expected-winner = (index-of-max expected-outputs)
       for outputs = (infer-frame net inputs)
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
       for outputs = (infer-frame net inputs)
       for frame-count = 1 then (1+ frame-count)
       for frame-error = (loop for actual in outputs
                            for expected in expected-outputs
                            summing (expt (- expected actual) 2))
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
     for outputs = (infer-frame net inputs)
     for index = 0 then (1+ index)
     for frame-error = (loop for actual in outputs
                          for expected in expected-outputs
                          summing (expt (- expected actual) 2))
     do (setf (aref frame-errors index) frame-error)
     finally (return (average frame-errors))))

(defmethod infer ((net t-net) (frames list))
  (loop for frame in frames collect (infer-frame net frame)))

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
		 maximizing count into width
     do (push-tail (layer-dlist net) layer)
     finally (return (adjustments net))))

(defun adjustments (net)
	(loop for layer-node = (head (layer-dlist net)) then (next layer-node)
		 while layer-node
		 for layer = (value layer-node)
		 for layer-neuron-count = (len layer)
		 count layer into layer-count
		 maximize layer-neuron-count into width
		 sum layer-neuron-count into neuron-count
		 sum (cx-count layer) into cx-count
		 finally 
			 (name-neurons net)
			 (set-weight-range net)
			 (setf (layer-count net) layer-count
						 (width net) width
						 (neuron-count net) neuron-count
						 (cx-count net) cx-count)
			 (place-neurons net)
			 (return net)))

(defgeneric cx-count (thing)
	(:method ((layer dlist))
		(loop for neuron-node = (head layer) then (next neuron-node)
			 while neuron-node 
			 sum (loop with neuron = (value neuron-node)
							for cx-node = (head (cx-dlist neuron)) then (next cx-node)
							while cx-node count cx-node)))
	(:method ((net t-net))
		(loop for layer-node = (head (layer-dlist net)) then (next layer-node)
			 while layer-node sum (cx-count (value layer-node)))))

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
       (funcall (connect-function net) net)
       (reset-weights net)
       (create-gates net)
       (setf (log-file net) (default-log-file-name net)
						 (weights-file net) (default-weights-file-name net))
       (return (adjustments net))))
                               
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
						 (let ((name (format nil "~a-~a~a" 
																 layer-index
																 neuron-index
																 (if (biased neuron) "b" ""))))
							 (setf (name neuron) name)
							 (setf (gethash name (neuron-index net)) neuron)))))

(defun place-neurons (net)
	(loop with height = 1.0
		 and width = 1.0
		 and top-margin = 0.05
		 and bottom-margin = 0.05
		 and left-margin = 0.05
		 and right-margin = 0.05
		 with y-spacing = (/ (- height top-margin bottom-margin) (1- (layer-count net)))
		 for layer-node = (head (layer-dlist net)) then (next layer-node)
		 while layer-node
		 for layer = (value layer-node)
		 for y = top-margin then (+ y y-spacing)
		 for x-spacing = (/ (- width left-margin right-margin) (len layer))
		 do (loop for neuron-node = (head layer) then (next neuron-node)
					 while neuron-node
					 for neuron = (value neuron-node)
					 for x = (+ left-margin (/ x-spacing 2)) then (+ x x-spacing)
					 do (setf (x-coor neuron) x
										(y-coor neuron) y))))

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
                      :momentum momentum)
		(adjustments net)))

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
     finally (return (loop for chars in (reverse (cons (reverse word) (reverse words)))
                        for first = t then nil
                        for string = (map 'string 'identity chars)
                        collect (if first string (read-from-string string))))))

(defun join-paths (&rest path-parts)
  "Joins elements of PATH-PARTS into a file path, inserting slashes where necessary."
  (format nil "~a~{~a~^/~}"
          (if (scan "^/" (car path-parts)) "/" "")
          (loop for part in path-parts
             for first = t then nil
             for clean-part = (if (scan "//+$" part)
                                  (subseq part 0 (1- (length part)))
                                  (regex-replace-all "^/+|/+$" part ""))
             unless (or (null clean-part)
                        (zerop (length clean-part)))
             collect clean-part)))

(defun create-environment
    (id
     topology
     &key 
       (home-folder (join-paths (namestring (user-homedir-pathname))
                                "common-lisp" "dc-bianet"))
       (test-file "mnist-0-1-test.csv")
       (training-file "mnist-0-1-train.csv")
       (weight-reset-function (make-random-weight-fn :min -0.5 :max 0.5))
       (make-current t)
       (cx-mode :full)
       (cx-params 12)
       (learning-rate *default-learning-rate*)
       (momentum *default-momentum*))
  (let* ((training-file-name (join-paths home-folder training-file))
         (test-file-name (join-paths home-folder test-file))
         (label-counts (type-1-file->label-counts training-file-name))
         (label->index (label-counts->label-indexes label-counts))
         (label->expected-outputs (label-outputs-hash label->index))
         (index->label (loop for label being the hash-keys in label->index
                          collect label into list
                          finally
                            (return
                              (map 'vector 'identity
                                   (sort list
                                         (lambda (a b)
                                           (< (gethash a label->index)
                                              (gethash b label->index))))))))
         (training-set (file->training-set training-file-name
                                           label->expected-outputs))
         (test-set (file->test-set test-file-name
                                   label->expected-outputs))
         (net (create-standard-net
               topology
               :id id
               :weight-reset-function weight-reset-function
               :cx-mode cx-mode
               :cx-params cx-params
               :learning-rate learning-rate
               :momentum momentum))
         (environment (make-instance 't-environment
                                     :id id
                                     :net net
                                     :training-file training-file-name
                                     :test-file test-file-name
                                     :training-set training-set
                                     :test-set test-set
                                     :label->index label->index
                                     :label->expected-outputs label->expected-outputs
                                     :index->label index->label)))
    (setf (getf *environments* id) environment)
    (when make-current (set-current-environment id))
    environment))

;; (defun create-environment-from-pngs
;;     (id
;;      &key
;;        (png-folder (error "png-folder parameter is required."))
;;        hidden-layers
;;   (let* ((training-file-name (join-paths png-folder
;;                                          (format nil "~(~a~)-train.lisp" id)))
;;          (test-file-name (join-paths png-folder
;;                                      (format nil "~(~a~)-test.lisp" id)))
;;          (output-labels (relative-subdirectories-of png-folder))
;;          (input-count (length (read-png (any-png-in png-folder)))))
         

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
         (inputs (normalize-list (read-png file)))
         (outputs (gethash label (label->expected-outputs environment))))
    (list inputs outputs)))

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

(defun train (id &key
                   (epochs 100)
                   (target-error 0.05)
                   (reset-weights t)
                   (thread-count *default-thread-count*)
                   (report-function #'plotting-report-function)
                   (report-frequency 10)
                   plot-errors
                   weights-file
                   skip-refresh)
  (let ((environment (getf *environments* id)))
    (when (not environment) (error "No such environment ~(~a~)" id))
    (when (or reset-weights (not weights-file))
      (clear (training-error environment)))
    (when weights-file
      (apply-weights-from-file (net environment) weights-file)
      (setf reset-weights nil))
    (setf (plot-errors environment) plot-errors)
    (when (plot-errors environment)
      (title (format nil "~(~a~) ~{~a~^-~} Training Error"
                     id (simple-topology (net environment))))
      (axis (list t t 0 1.0)))
    (start-thread-pool thread-count)
    (train-frames environment
                  (training-set environment)
                  #'training-complete
                  :epochs epochs
                  :target-error target-error
                  :reset-weights reset-weights
                  :report-frequency report-frequency
                  :report-function report-function
                  :skip-refresh skip-refresh))
  :training)

(defun set-current-environment (id)
  (unless (setf *environment* (environment-by-id id))
    (error "No environment for key ~(~a~)." id))
  (setf *net* (net *environment*)
        *training-set* (training-set *environment*)
        *test-set* (test-set *environment*))
  *environment*)

(defun set-weight-range (net)
	(loop with max-weight = nil and min-weight = nil
		 for layer-node = (head (layer-dlist net)) then (next layer-node)
		 while layer-node
		 do (loop with layer = (value layer-node) 
					 for neuron-node = (head layer) then (next neuron-node)
					 while neuron-node
					 for neuron = (value neuron-node)
					 do (loop for cx-node = (head (cx-dlist neuron)) then (next cx-node)
								 while cx-node
								 for cx = (value cx-node)
								 for weight = (weight cx)
								 when (or (null max-weight) (> weight max-weight))
								 do (setf max-weight weight)
								 when (or (null min-weight) (< weight min-weight))
								 do (setf min-weight weight)))
		 finally 
			 (setf (max-weight net) max-weight)
			 (setf (min-weight net) min-weight)))

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
		(set-weight-range (net environment))
    (collect-weights-into-file (net environment))))

(defun wait-for-training-completion (id)
  "Waits for training to complete, returns (list fitness% elapsed-seconds presentation network-error)"
  (let ((environment (getf *environments* id)))
    (loop while (get-training-in-progress) do (sleep 1)
       finally (return (cons (getf (fitness environment) :percent)
                             (value (tail (training-error environment))))))))
  
(defun shuffle (seq)
  "Return a sequence with the same elements as the given sequence S, but in random order (shuffled)."
  (loop
     with l = (length seq) 
     with w = (make-array l :initial-contents seq)
     for i from 0 below l 
     for r = (random l) 
     for h = (aref w i)
     do 
       (setf (aref w i) (aref w r)) 
       (setf (aref w r) h)
     finally (return (if (listp seq) (map 'list 'identity w) w))))

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

(defun read-png (filename &key (width 28) (height 28))
  (loop with image-data = (png-read:image-data
                           (png-read:read-png-file filename))
     with dimensions = (length (array-dimensions image-data))
     for y from 0 below height appending
       (loop for x from 0 below width collecting
            (if (= dimensions 2)
                (aref image-data x y)
                (aref image-data x y 0)))
     into intensity-list
     finally (return (invert-intensity intensity-list))))

(defun invert-intensity (list &key (max 255))
  (loop for element in list collect (- max element)))

(defun normalize-list (list &key (max 255) (min 0))
  (loop with range = (- max min)
     for element in list collect (float (+ (/ element range) min))))

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

(defun denormalize-list (inputs &key (min 0) (max 255))
  (loop with size = (- max min)
     for input in inputs collect (truncate (+ (* input size) min))))

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
     for normalized-file-data = (normalize-list (read-png file))
     for outputs = (infer-frame *net* normalized-file-data)
     for label = (outputs->label *environment* outputs)
     collect (list (file-namestring file) label)))

(defun train-on-png (id label file count)
  (loop with environment = (environment-by-id id)
     with net = (net environment)
     with frame = (png-file->frame :digits label file)
     with inputs = (car frame)
     with expected-outputs = (second frame)
     for a from 1 to count do 
       (train-frame net inputs expected-outputs)
     finally (return (outputs->label environment (infer-frame net inputs)))))

(defun relative-subdirectories-of (path)
  (loop with dir-spec = (format nil "~a~a*" path (if (scan "/$" path) "" "/"))
     for subdirectory in (directory dir-spec)
     when (uiop:directory-pathname-p subdirectory)
     collect (let* ((path (directory-namestring subdirectory)))
               (pathname-name (subseq path 0 (1- (length path)))))
     into relative-subdirectories
     finally (return (sort relative-subdirectories #'string<))))

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
  (loop with dir-spec = (format nil "~a/*.png" label-folder)
     for file in (directory dir-spec)
     for inputs = (normalize-list (read-png file))
     collect (list inputs expected-outputs) into frames
     finally (return (if as-vector
                         (map 'vector 'identity frames)
                         frames))))

(defun pngs->frames (png-tree-path &key as-vector)
  (loop with labels = (relative-subdirectories-of png-tree-path)
     with label->index = (list->key-index labels)
     with label->expected-outputs = (label-outputs-hash label->index)
     for label in labels
     for label-folder = (join-paths png-tree-path label)
     for expected-outputs = (gethash label label->expected-outputs)
     appending (pngs->frames-for-label label-folder expected-outputs)
     into frames
     finally (return (if as-vector
                         (map 'vector 'identity frames)
                         frames))))

(defun any-png-in (png-tree-path)
  (let* ((output-labels (relative-subdirectories-of png-tree-path))
         (dir-spec (join-paths png-tree-path (car output-labels) "*.png"))
         (files (directory dir-spec)))
    (car files)))

(defun pngs->suggest-topology (png-tree-path)
  (loop with output-count = (length (relative-subdirectories-of png-tree-path))
     with input-count = (length (read-png (any-png-in png-tree-path)))
     for power = 1 then (1+ power)
     while (< (expt 2 power) output-count)
     finally (return (list input-count (expt 2 (1+ power)) output-count))))


(defun evaluate-topologies (&key
                              (id :zero-or-one)
                              (inputs 784)
                              (outputs 2)
                              (hidden-start 16)
                              (hidden-stop 32)
                              (hidden-step 16)
                              (training-file "mnist-0-1-train.csv")
                              (test-file "mnist-0-1-test.csv")
                              (init-weights-function (make-random-weight-fn))
                              (report-frequency 1))
  (loop with row-format = "|~{ ~a | ~}"
     for hidden from hidden-start to hidden-stop by hidden-step
     for environment = (create-environment
                        id (list inputs hidden outputs)
                        :training-file training-file
                        :test-file test-file
                        :weight-reset-function init-weights-function)
     for training = (progn (train id :report-frequency report-frequency)
                           (format t "~a~%" (log-file *net*)))
     for (fitness elapsed presentations network-error) =
       (wait-for-training-completion id)
     do (format t "fit=~,2f%; hidden=~d; secs=~d; presented=~d; error=~d~%"
                fitness hidden elapsed presentations network-error)
     collect (format nil row-format
                     (list fitness hidden elapsed presentations network-error))
     into lines
     finally (push
              (format nil row-format
                      '("fitness" "hidden" "elapsed" "presentations"
                        "network-error"))
              lines)
       (return (format nil "~{~a~%~}" lines))))

(defun evaluate-convergence-variance (&key
                                        (id :zero-or-none)
                                        (inputs 784)
                                        (outputs 2)
                                        (hidden-units 16)
                                        (iterations 5)
                                        (training-file "mnist-0-1-train.csv")
                                        (test-file "mnist-0-1-test.csv")
                                        (init-weights-function (make-random-weight-fn))
                                        (report-frequency 1))
  (loop with row-format = "|~{ ~a | ~}"
     for iteration from 1 to iterations
     for environment = (create-environment
                        id (list inputs hidden-units outputs)
                        :training-file training-file
                        :test-file test-file
                        :weight-reset-function init-weights-function)
     for training = (progn (train id :report-frequency report-frequency)
                           (format t "~a~%" (log-file *net*)))
     for (fitness elapsed presentations network-error) =
       (wait-for-training-completion id)
     do (format t "fit=~,2f%; hidden=~d; secs=~d; presented=~d; error=~d~%"
                fitness hidden-units elapsed presentations network-error)
     collect (format nil row-format
                     (list fitness hidden-units elapsed presentations network-error))
     into lines
     finally (push
              (format nil row-format
                      '("fitness" "hidden" "elapsed" "presentations"
                        "network-error"))
              lines)
       (return (format nil "~{~a~%~}" lines))))
                                        
;;; clim

(defparameter *bianet-frame* nil)

(define-application-frame bianet ()
	((net :accessor net :initarg :net :initform nil)
	 (zoom :accessor zoom :initarg :zoom :initform 100)
	 (radius :accessor radius :initarg :radius :initform 2)
	 (hi-pass :accessor hi-pass :initarg :hi-pass :initform 0.8)
	 (lo-pass :accessor lo-pass :initarg :lo-pass :initform 0.2))
	(:geometry :width 1200 :height 800)
	(:menu-bar t)
	(:panes
	 (network :application :display-time t :display-function #'render-neural-network)
	 (neuron :application :width 300)
	 (training-error :application)
	 (input :interactor :width 400))
	(:layouts (default
								(vertically ()
									(4/8 (horizontally ()
												 (:fill network)
												 neuron))
									(2/8 training-error)
									(2/8 input)))))

(defun render-neural-network (frame pane)
	(loop with net = (net frame)
		 and width = (bounding-rectangle-width pane)
		 and height = (bounding-rectangle-height pane)
		 with min-weight = (min-weight net)
		 with weight-span = (- (max-weight net) min-weight)
		 initially (window-clear pane)
		 for layer-node = (head (layer-dlist net)) then (next layer-node)
		 while layer-node do
			 (loop with layer = (value layer-node) 
					for neuron-node = (head layer) then (next neuron-node)
					while neuron-node
					for neuron = (value neuron-node)
					for x = (truncate (* width (x-coor neuron) (/ (zoom frame) 100)))
					for y = (truncate (* height (y-coor neuron) (/ (zoom frame) 100)))
					do (draw-circle* pane x y (radius frame) 
													 :ink +black+ :line-thickness 1 :filled nil)
						(loop for cx-node = (head (cx-dlist neuron)) then (next cx-node)
							 while cx-node
							 for cx = (value cx-node)
							 for target-neuron = (target cx)
							 for target-x = (truncate (* width (x-coor target-neuron) (/ (zoom frame) 100)))
							 for target-y = (truncate (* height (y-coor target-neuron) (/ (zoom frame) 100)))
							 for luminosity = (/ (- (weight cx) min-weight) weight-span)
							 when (or (< luminosity (lo-pass frame)) (> luminosity (hi-pass frame)))
							 do (draw-line* pane x y target-x target-y 
															:ink (make-gray-color luminosity) :line-thickness 1)))
		 finally (format pane "~%~%")))

(define-bianet-command (com-zoom :menu t :name "Zoom")
		((zoom 'integer))
	(setf (zoom *bianet-frame*) zoom)
	(setf (pane-needs-redisplay (get-frame-pane *bianet-frame* 'network)) t))

(define-bianet-command (com-radius :menu t :name "Radius")
		((radius 'integer))
	(setf (radius *bianet-frame*) radius)
	(setf (pane-needs-redisplay (get-frame-pane *bianet-frame* 'network)) t))

(define-bianet-command (com-luminosity-filter :menu t :name "Luminosity Filter")
		((lo-pass 'float) (hi-pass 'float))
	(setf (lo-pass *bianet-frame*) lo-pass
				(hi-pass *bianet-frame*) hi-pass)
	(setf (pane-needs-redisplay (get-frame-pane *bianet-frame* 'network)) t))

(defun run (net)
	(setf *bianet-frame* (make-application-frame 'bianet :net net))
	(make-thread (lambda () (run-frame-top-level *bianet-frame*))
							 :name "bianet-frame"))
