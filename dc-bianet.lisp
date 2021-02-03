(in-package :dc-bianet)

(defparameter *magnitude-limit* 1e9)
(defparameter *precision-limit* 1e-9)
(defparameter *default-learning-rate* 0.02)
(defparameter *default-momentum* 0.1)
(defparameter *default-min-weight* -0.9)
(defparameter *default-max-weight* 0.9)
(defparameter *job-queue* nil) ;; Instance of dlist
(defparameter *thread-pool-running* nil)
(defparameter *job-counter* 0)
(defparameter *job-counter-mutex* (make-mutex :name "job-counter-mutex"))
(defparameter *gate* (make-gate))
(defparameter *thread-pool* nil)
(defparameter *jobs-run* nil) ;; Instance of dlist
(defparameter *target-input-impact-mutex* (make-mutex :name "target-input-impact"))
(defparameter *gates* nil)

(defun thread-work ()
  (loop while *thread-pool-running*
     for job = (when *job-queue* (pop-tail *job-queue*))
     do (case (car job)
          ((:fire :backprop)
           (funcall (second job))
           (push-tail *jobs-run* (first job))
           (with-mutex (*job-counter-mutex*) (incf *job-counter*)))
          (:open-gate 
           (open-gate (elt *gates* (second job)))))))

(defun start-thread-pool (thread-count)
  (setf *thread-pool-running* t)
  (setf *job-queue* (make-instance 'dlist))
  (setf *jobs-run* (make-instance 'dlist))
  (setf *job-counter* 0)
  (setf *thread-pool*
        (loop for a from 1 to thread-count collect
             (make-thread #'thread-work :name "thread-work"))))

(defun get-job-count ()
  (with-mutex (*job-counter-mutex*)
    *job-counter*))

(defun stop-thread-pool ()
  (setf *thread-pool-running* nil)
  (when *thread-pool*
    (loop for thread in *thread-pool* do (join-thread thread))
    *thread-pool* nil))

(defun make-random-weight-fn (&key (min 0.0) (max 1.0))
  (lambda (rstate &key
                    layer-index
                    source-index
                    target-index
                    global-source-index
                    global-target-index)
    (declare (ignore layer-index source-index target-index 
                     global-source-index global-target-index))
    (+ min (random (- max min) rstate))))

(defun make-limiter (&key (magnitude *magnitude-limit*)
                       (precision *precision-limit*))
  (lambda (x)
    (let ((limited-precision (if (< (abs x) precision)
                               (* (signum x) precision)
                               x)))
    (if (> (abs limited-precision) magnitude)
        (* (signum limited-precision) magnitude)
        limited-precision))))

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
  (cond ((> x 16.64) 1.0)
        ((< x -88.7) 0.0)
        ((< (abs x) 1e-8) 0.5)
        (t (/ 1.0 (1+ (exp (- x)))))))

(defun logistic-derivative (x)
  (* x (- 1 x)))

(defun relu (x)
  (max 0 x))

(defun relu-derivative (x)
  (if (<= x 0.0) 0.0 1.0))

(defun relu-leaky (x)
  (max 0 x))

(defun relu-leaky-derivative (x)
  (if (<= x 0.0) 0.001 1.0))
                              
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
   (weight :accessor weight :initarg :weight :initform 0.1 :type real)
   (learning-rate :accessor learning-rate :initarg :learning-rate :type real
                  :initform 0.02)
   (momentum :accessor momentum :initarg :momentum :initform 0.1)
   (delta :accessor delta :initarg :delta :initform 0.0)
   (limiter :accessor limiter :initarg :limiter
            :initform (make-limiter))
   (weight-mtx :reader weight-mtx :initform (make-mutex))))

(defclass t-neuron ()
  ((id :accessor id :initarg :id :type keyword :initform (bianet-id))
   (name :accessor name :initarg :name :type string :initform nil)
   (input :accessor input :type real :initform 0.0)
   (biased :accessor biased :initarg :biased :type boolean :initform nil)
   (transfer-key :accessor transfer-key :initarg :transfer-key 
                 :initform :logistic)
   (transfer-function :accessor transfer-function :type function)
   (transfer-derivative :accessor transfer-derivative :type function)
   (output :accessor output :type real :initform 0.0)
   (expected-output :accessor expected-output :type real :initform 0.0)
   (err :accessor err :type real :initform 0.0)
   (err-derivative :accessor err-derivative :type real :initform 0.0)
   (x-coor :accessor x-coor :type real :initform 0.0)
   (y-coor :accessor y-coor :type real :initform 0.0)
   (z-coor :accessor z-coor :type real :initform 0.0)
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
   (log-file :accessor log-file :initarg :log-file :type string)
   (stop-training :accessor stop-training :type boolean :initform nil)
   (rstate :reader rstate :initform (make-random-state))))

(defgeneric feedforward (thing)

  (:method ((net t-net))
    (loop
       for layer-node = (head (layer-dlist net)) then (next layer-node)
       while layer-node
       for layer-index = 0 then (1+ layer-index)
       for gate = (elt *gates* layer-index) 
       do 
         (close-gate gate)
         (feedforward (value layer-node))
         (push-head *job-queue* (list :open-gate layer-index))
         (wait-on-gate gate)))

  (:method ((layer dlist))
    (loop 
       for neuron-node = (head layer) then (next neuron-node)
       while neuron-node
       do (push-head *job-queue* 
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
       do (backpropagate (value layer-node))))

  (:method ((layer dlist))
    (loop
       for neuron-node = (tail layer) then (prev neuron-node)
       while neuron-node 
       do (push-head *job-queue* 
                     (list :backprop
                           (let ((neuron (value neuron-node)))
                             (lambda () (backpropagate neuron)))))))

  (:method ((neuron t-neuron))
    (compute-neuron-error neuron)
    (adjust-neuron-cx-weights neuron)))

(defgeneric backpropagate (thing)
  (:method ((net t-net))
    (loop
       for layer-node = (tail (layer-dlist net)) then (prev layer-node)
       while layer-node
       do (backpropagate (value layer-node))))

  (:method ((layer dlist))
    (loop
       for neuron-node = (tail layer) then (prev neuron-node)
       while neuron-node do (backpropagate (value neuron-node))))

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
               summing (* (weight cx) (err-derivative (target cx)))))))
    (with-mutex ((err-mtx neuron)) (setf (err neuron) err))
    (with-mutex ((err-der-mtx neuron))
      (setf (err-derivative neuron)
            (* err (funcall (transfer-derivative neuron) (output neuron)))))))
        

(defmethod adjust-neuron-cx-weights ((neuron t-neuron))
    (loop
       with cx-dlist = (cx-dlist neuron)
       for cx-node = (head cx-dlist) then (next cx-node)
       while cx-node do (adjust-cx-weight (value cx-node))))

;; (defmethod adjust-cx-weight ((cx t-cx))
;;   (let* ((delta (* (learning-rate cx)
;;                    (err-derivative (target cx))
;;                    (output (source cx))))
;;          (new-weight (funcall (limiter cx)
;;                               (+ (weight cx) delta (* (momentum cx) (delta cx))))))
;;     (setf (weight cx) new-weight)))

(defmethod adjust-cx-weight ((cx t-cx))
  (let* ((delta (* (learning-rate cx)
                   (err-derivative (target cx))
                   (output (source cx))))
         (new-weight (+ (weight cx) delta (* (momentum cx) (delta cx)))))
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
          (setf (input neuron) input-value))))

(defmethod apply-outputs ((net t-net) (output-values list))
  (loop with output-layer-node = (tail (layer-dlist net))
     with output-layer = (value output-layer-node)
     for neuron-node = (head output-layer) then (next neuron-node)
     while neuron-node
     for neuron = (value neuron-node)
     for output-value in output-values
     do (setf (output neuron) output-value)))

(defmethod apply-expected-outputs ((net t-net) (expected-output-values list))
  (loop with layer-dlist = (value (tail (layer-dlist net)))
     for neuron-node = (head layer-dlist) then (next neuron-node)
     while neuron-node
     for neuron = (value neuron-node)
     for expected-output-value in expected-output-values
     do (setf (expected-output neuron) expected-output-value)))

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

(defgeneric apply-weights (thing weights)
  (:method ((net t-net) (weights list))
    (loop for cx in (collect-cxs net)
       for weight in weights
       do (setf (weight cx) weight
                (delta cx) 0.0))))

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
  (apply-inputs net inputs)
  (feedforward net)
  (collect-outputs net))

(defmethod train-frame ((net t-net) (inputs list) (expected-outputs list))
  (infer-frame net inputs)
  (apply-expected-outputs net expected-outputs)
  (backpropagate net))

(defun train-frames (net training-frames 
                     &key
                       (epochs 6)
                       (target-error 0.05)
                       (randomize-weights t)
                       (report-function #'default-report-function)
                       (report-frequency #'default-report-frequency))
  (when randomize-weights (randomize-weights net))
  (loop 
     with start-time = (get-universal-time)
     with last-report-time = start-time
     with error-set-count = (cond ((> (length training-frames) 10000)
                                      (truncate (* (length training-frames) 0.1)))
                                     ((> (length training-frames) 1000)
                                      1000)
                                     (t (length training-frames)))
     with error-set = (choose-from-vector training-frames error-set-count)
     for indexes = (shuffle (loop for a from 0 below (length training-frames) collect a))
     for epoch from 1 to epochs
     for network-error = (network-error *net* error-set)
     while (> network-error target-error)
     do (loop 
           for index in indexes
           for (inputs expected-outputs) = (aref training-frames index)
           for count = 1 then (1+ count)
           for elapsed-seconds = (- (get-universal-time) start-time)
           for since-last-report = (- (get-universal-time) last-report-time)
           do (train-frame net inputs expected-outputs)
           when (funcall report-frequency count elapsed-seconds since-last-report)
           do (funcall report-function
                       count 
                       elapsed-seconds
                       (network-error *net* error-set))
             (setf last-report-time (get-universal-time)))
     finally (return network-error)))

(defun default-report-frequency (count elapsed-seconds since-last-report)
  (declare (ignore count elapsed-seconds))
  (>= since-last-report 10))

(defun default-report-function (count elapsed-seconds network-error)
  (format t "elapsed=~d; processed=~d; error=~d~%" 
          elapsed-seconds count network-error))

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

;; Type-1 files are CSV files that have rows with one label at the
;; beginning of the row, followed by input values.  You can determine
;; the number of output neurons needed by counting the distinct labels
;; present in the file.
(defun train-on-type-1-file (net filename)
  (let ((label-counts (type-1-file->label-counts filename)))
    (with-open-file (file filename)
      (loop with label-outputs = (label-outputs-hash
                                   (label-counts->label-indexes label-counts))
         for line = (read-line file nil)
         for line-number = 1 then (1+ line-number)
         while (and line (> (length line) 1))
         for values = (split "," line)
         for label = (car values)
         for inputs = (mapcar #'read-from-string (cdr values))
         for expected-outputs = (gethash label label-outputs)
         do (train-frame net inputs expected-outputs)
         when (zerop (mod line-number 100))
         do (format t "processed ~d~%" line-number)))))

(defun type-1-file->set (filename)
  (let* ((label-counts (type-1-file->label-counts filename))
         (label-outputs (label-outputs-hash 
                        (label-counts->label-indexes label-counts)))
         (line-count (file-line-count filename))
         (set (make-array line-count :element-type 'list :initial-element nil)))
    (with-open-file (file filename)
      (loop for line = (read-line file nil)
         for index = 0 then (1+ index)
         while (and line (> (length line) 1))
         for values = (type-1-csv-line->label-and-inputs line)
         for label = (car values)
         for inputs = (cdr values)
         for expected-outputs = (gethash label label-outputs)
         do (setf (aref set index) (list inputs expected-outputs))))
    set))
         
(defun label-outputs-hash (label-indexes)
  (loop with label-outputs = (make-hash-table :test 'equal)
     for label being the hash-keys in label-indexes using (hash-value index)
     do (setf (gethash label label-outputs)
              (label->outputs label label-indexes))
     finally (return label-outputs)))

(defun label->outputs (label label-indexes)
  (loop with index = (gethash label label-indexes)
     for a from 0 below (hash-table-count label-indexes)
     collect (if (= a index) 1.0 0.0)))

(defun outputs->label (outputs index-labels)
  (gethash (index-of-max outputs) index-labels))
         
(defun label-counts->label-indexes (label-counts)
  (loop with label-indexes = (make-hash-table :test 'equal)
     for label in (sort (loop for label being the hash-keys in label-counts
                           collect label)
                        #'string<)
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

(defmethod index-of-max ((list list))
 (loop with max-index = 0 and max-value = (elt list 0)
       for value in list
       for index = 0 then (1+ index)
       when (> value max-value)
       do 
         (setf max-index index)
         (setf max-value value)
       finally (return max-index)))

(defgeneric evaluate-inference-1hs (net training-frames)
  (:method ((net t-net) (training-frames list))
    (loop 
       for (inputs expected-outputs) in training-frames
       for index = 0 then (1+ index)
       for expected-winner = (index-of-max expected-outputs)
       for outputs = (infer-frame net inputs)
       for winner = (index-of-max outputs)
       for total = 1 then (1+ total)
       for pass = (= winner expected-winner)
       for correct = (if pass 1 0) then (if pass (1+ correct) correct)
       finally 
         (return (list :percent 
                       (float (* 100 (/ correct total)))
                       :total total 
                       :pass correct 
                       :fail (- total correct)))))
  (:method ((net t-net) (training-frames array))
    (evaluate-inference-1hs net (map 'list 'identity training-frames))))
  
(defmethod network-error ((net t-net) (frames list))
  (loop 
     for (inputs expected-outputs) in frames
     for outputs = (infer-frame net inputs)
     for frame-count = 1 then (1+ frame-count)
     for frame-error = (loop for actual in outputs
                          for expected in expected-outputs
                          summing (expt (- expected actual) 2))
     summing frame-error into total-error
     finally (return (/ total-error frame-count))))

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
  (loop with log = (or log-file (format nil "/tmp/~(~a~).log" id))
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
                              log-file
                              (weight-reset-function 
                               (make-random-weight-fn :min -0.9 :max 0.9))
                              (limiter (make-limiter))
                              (momentum *default-momentum*)
                              (learning-rate *default-learning-rate*))
  (loop with log = (or log-file (format nil "/tmp/~(~a~).log" id))
     with net = (make-instance 't-net :id id :log-file log)
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
       (connect-fully net
                      :learning-rate learning-rate
                      :momentum momentum
                      :initial-weight-function weight-reset-function
                      :limiter limiter)
       (create-gates net)
       (return net)))
                               
(defmethod create-gates ((net t-net))
  (setf *gates* 
        (loop for layer-node = (head (layer-dlist net)) then (next layer-node)
           while layer-node
           for layer-index = 0 then (1+ layer-index)
           for gate-name = (format nil "~(~a-~d~)" (id net) layer-index)
           collecting (make-gate :name gate-name))))

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
                            (momentum *default-momentum*)
                            (initial-weight-function 
                             (make-random-weight-fn :min -0.9 :max 0.9))
                            (limiter (make-limiter)))
  (loop with layer-count = (len (layer-dlist net)) 
     and global-source-index = 0
     and global-target-index = 0
     for layer-node = (head (layer-dlist net)) then (next layer-node)
     for layer-index = 1 then (1+ layer-index)
     while (< layer-index layer-count)
     for layer = (value layer-node)
     for next-layer = (value (next layer-node))
     do (loop for source-node = (head layer) then (next source-node)
           for source-index = 1 then (1+ source-index)
           while source-node do 
             (incf global-source-index)
             (loop for target-node = (head next-layer) then (next target-node)
                 for target-index = 0 then (1+ target-index)
                 while target-node
                 for source = (value source-node)
                 for target = (value target-node)
                 when (not (biased target))
                 do
                   (incf global-target-index)
                   (push-tail 
                     (cx-dlist source)
                     (make-instance 
                      't-cx 
                      :source source 
                      :target target
                      :learning-rate learning-rate
                      :momentum momentum
                      :limiter limiter
                      :weight (funcall 
                               initial-weight-function
                               (rstate net)
                               :layer-index layer-index
                               :source-index source-index
                               :target-index target-index
                               :global-source-index global-source-index
                               :global-target-index global-target-index)))))))

(defgeneric randomize-weights (thing &key min max)
  (:method ((net t-net) &key min max)
    (loop 
       with a = (or min *default-min-weight*) 
       and b = (or max *default-max-weight*)
       for layer-node = (head (layer-dlist net)) then (next layer-node)
       while layer-node
       do (randomize-weights (value layer-node) :min a :max b)))
  (:method ((layer dlist) &key min max)
    (loop 
       with a = (or min *default-min-weight*) 
       and b = (or max *default-max-weight*)
       for neuron-node = (head layer) then (next neuron-node)
       while neuron-node
       do (randomize-weights (value neuron-node) :min a :max b)))
  (:method ((neuron t-neuron) &key min max)
    (loop 
       with a = (or min *default-min-weight*) 
       and b = (or max *default-max-weight*)
       for cx-node = (head (cx-dlist neuron)) then (next cx-node)
       while cx-node
       do (randomize-weights (value cx-node) :min a :max b)))
  (:method ((cx t-cx) &key min max)
    (let ((a (or min *default-min-weight*))
          (b (or max *default-max-weight*)))
      (setf (weight cx) (+ (random (- b a)) a)))))
           

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

(defparameter *frames-train* nil)
(defparameter *frames-test* nil)
(defparameter *net* nil)

(defun test-1-setup ()
  (let* ((folder "/home/macnod/google-drive/dc/cloud-local/Projects/Mindrigger/data/mnist")
         (file-train (format nil "~a/~a" folder "mnist-0-1-train.csv"))
         (file-test (format nil "~a/~a" folder "mnist-0-1-test.csv")))
    (setf *frames-train* (map 'vector 'identity 
                              (normalize-set (type-1-file->set file-train))))
    (setf *frames-test* (normalize-set (type-1-file->set file-test)))
    (setf *net* (create-standard-net '(784 16 2) :id :test-1))))

(defun test-2-setup ()
  (let* ((folder "/home/macnod/google-drive/dc/cloud-local/Projects/Mindrigger/data/mnist")
         (file-train (format nil "~a/~a" folder "mnist-train.csv"))
         (file-test (format nil "~a/~a" folder "mnist-test.csv")))
    (setf *frames-train* (map 'vector 'identity
                              (normalize-set (type-1-file->set file-train))))
    (setf *frames-test* (normalize-set (type-1-file->set file-test)))
    (setf *net* (create-standard-net '(784 128 10) :id :test-1))))

(defun test-1-infer-d ()
  (start-thread-pool 8)
  (let ((result (infer-frame *net* (car (car *frames-test*)))))
    (stop-thread-pool)
    result))

(defun test-1-infer ()
  (infer-frame *net* (car (car *frames-test*))))

(defun test-train (&key (epochs 6) (thread-count 8) (randomize-weights t))
  (stop-thread-pool)
  (start-thread-pool thread-count)
  (let ((start-time (get-internal-real-time))
        (network-error (train-frames *net* *frames-train* 
                                     :epochs epochs 
                                     :target-error 0.05 
                                     :randomize-weights randomize-weights))
        (stop-time (get-internal-real-time)))
    (stop-thread-pool)
    (list :time (/ (- stop-time start-time) 1000.0)
          :network-error network-error
          :result (evaluate-inference-1hs *net* *frames-test*))))

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
