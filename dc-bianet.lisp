(in-package :dc-bianet)

(defparameter *magnitude-limit* 1e9)
(defparameter *precision-limit* 1e-9)
(defparameter *default-learning-rate* 0.02)
(defparameter *default-momentum* 0.1)
(defparameter *default-min-weight* -0.9)
(defparameter *default-max-weight* 0.9)

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
            :initform (make-limiter))))

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
   (cx-dlist :accessor cx-dlist :type dlist :initform (make-instance 'dlist))))

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
    (setf (output neuron) output
          (input neuron) (if (not biased) 0.0 input)
          (err neuron) nil)))

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
       do (feedforward (value layer-node))))

  (:method ((layer dlist))
    (loop 
       for neuron-node = (head layer) then (next neuron-node)
       while neuron-node
       do (feedforward (value neuron-node))))

  (:method ((neuron t-neuron))
    (loop initially (transfer neuron)
       for cx-node = (head (cx-dlist neuron)) then (next cx-node)
       while cx-node
       do (feedforward (value cx-node))))

  (:method ((cx t-cx))
    (incf (input (target cx)) 
          (* (weight cx) (output (source cx))))))

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
  (setf (err neuron)
        (if (zerop (len (cx-dlist neuron)))
            ;; This is an output neuron (no outgoing connections)
            (- (expected-output neuron) (output neuron))
            ;; This is an input-layer or hidden-layer neuron;  we need 
            ;; to use the errors of downstream neurons to compute the
            ;; error of this neuron
            (loop
               for cx-node = (head (cx-dlist neuron)) then (next cx-node)
               while cx-node
               for cx = (value cx-node)
               summing (* (weight cx) (err-derivative (target cx))))))
  (setf (err-derivative neuron)
        (* (err neuron)
           (funcall (transfer-derivative neuron) (output neuron)))))

(defmethod adjust-neuron-cx-weights ((neuron t-neuron))
    (loop
       with cx-dlist = (cx-dlist neuron)
       for cx-node = (head cx-dlist) then (next cx-node)
       while cx-node do (adjust-cx-weight (value cx-node))))

(defmethod adjust-cx-weight ((cx t-cx))
  (let* ((delta (* (learning-rate cx)
                   (err-derivative (target cx))
                   (output (source cx))))
         (new-weight (funcall (limiter cx)
                              (+ (weight cx) delta (* (momentum cx) (delta cx))))))
    (setf (weight cx) new-weight)))

(defmethod apply-inputs ((net t-net) (input-values list))
  (loop with input-layer-node = (head (layer-dlist net))
     with input-layer = (value input-layer-node)
     for neuron-node = (head input-layer) then (next neuron-node)
     while neuron-node
     for neuron = (value neuron-node)
     for input-value in input-values
     do (setf (input neuron) input-value)))

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

(defgeneric infer-frames (net input-frames)
  (:method ((net t-net) (input-frames list))
    (loop for frame in input-frames collect (infer-frame net frame))))

(defmethod train-frames ((net t-net) (training-frames list))
  (loop for (inputs expected-outputs) in training-frames
     for count = 1 then (1+ count)
     do (train-frame net inputs expected-outputs)))

(defmethod index-of-max ((list list))
 (loop with max-index = 0 and max-value = (elt list 0)
       for value in list
       for index = 0 then (1+ index)
       when (> value max-value)
       do 
         (setf max-index index)
         (setf max-value value)
       finally (return max-index)))

(defmethod evaluate-inference-1hs ((net t-net) (training-frames list) 
                                   &key show-details)
  (loop 
     for (inputs expected-outputs) in training-frames
     for index = 0 then (1+ index)
     for expected-winner = (index-of-max expected-outputs)
     for outputs = (infer-frame net inputs)
     for winner = (index-of-max outputs)
     for total = 1 then (1+ total)
     for pass = (= winner expected-winner)
     for correct = (if pass 1 0) then (if pass (1+ correct) correct)
     when show-details
     collect (list :i index 
                   :in (mapcar #'display-float inputs)
                   :out (mapcar #'display-float outputs)
                   :exp (mapcar #'display-float expected-outputs))
     into details
     finally 
       (return 
         (let ((result (list :percent 
                             (format nil "~$%" (* 100 (/ correct total)))
                             :total total 
                             :pass correct 
                             :fail (- total correct))))
           (when show-details
             (setf (getf result :details) details))
           result))))
  
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

(defun create-standard-net (topology 
                            &key 
                              (transfer-function :relu)
                              (id (bianet-id))
                              log-file
                              (weight-reset-function 
                               (make-random-weight-fn :min -0.9 :max 0.9))
                              (limiter (make-limiter))
                              (momentum *default-momentum*)
                              (learning-rate *default-learning-rate*))
                              
  (loop with log = (or log-file

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
                             (make-random-weight-fn :min -0.9 :max 0.9)))
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

