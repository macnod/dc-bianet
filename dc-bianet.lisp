(in-package :dc-bianet)

(defparameter *magnitude-limit* 1e9)
(defparameter *precision-limit* 1e-9)
(defparameter *default-learning-rate* 0.02)
(defparameter *default-momentum* 0.1)

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

(defun limit-magnitude-and-precision (x)
  "Limit x to a positive and negative maximum and to a maximum precision when nearing 0."
  (let ((limited-precision (if (< (abs x) *precision-limit*)
                               (* (signum x) *precision-limit*)
                               x)))
    (if (> (abs limited-precision) *magnitude-limit*)
        (* (signum limited-precision) *magnitude-limit*)
        limited-precision)))

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
   (last-delta :accessor delta :initarg :delta :initform 0.0)
   (limiter :accessor limiter :initarg :limiter
            :initform #'limit-magnitude-and-precision)))

(defclass t-neuron ()
  ((id :accessor id :initarg :id :type keyword :initform (bianet-id))
   (input :accessor input :type real :initform 0.0)
   (biased :accessor biased :initarg :biased :type boolean :initform nil)
   (transfer-function :accessor transfer-function 
                      :initarg :transfer-function
                      :type function
                      :initform (getf (getf *transfer-functions* :logistic)
                                      :function))
   (transfer-derivative :accessor transfer-derivative
                        :initarg :transfer-derivative
                        :type function
                        :initform (getf (getf *transfer-functions* :logistic)
                                        :derivative))
   (output :accessor output :type real :initform 0.0)
   (expected-output :accessor expected-output :type real :initform 0.0)
   (err :accessor err :type real :initform 0.0)
   (err-derivative :accessor err-derivative :type real :initform 0.0)
   (x-coor :accessor x-coor :type real :initform 0.0)
   (y-coor :accessor y-coor :type real :initform 0.0)
   (z-coor :accessor z-coor :type real :initform 0.0)
   (cx-dlist :accessor cx-dlist :type dlist :initform (make-instance 'dlist))))

(defmethod initialize-instance :after ((neuron t-neuron) &key)
  (when (biased neuron)
    (setf (input neuron) 1.0)))

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
       while neuron-node
       do (backpropagate (value neuron-node))))

  (:method ((neuron t-neuron))
    (setf (err neuron)
          (if (zerop (len (cx-dlist neuron)))
              ;; This is an output neuron (no outgoing connections)
              (- (expected-output neuron) (output neuron))
              ;; This is an input-layer or hidden neuron
              (loop
                 for cx-node = (head (cx-dlist neuron)) then (next cx-node)
                 while cx-node
                 for cx = (value cx-node)
                 summing (* (weight cx) (err-derivative (target cx))))))
    (setf (err-derivative neuron)
          (funcall (transfer-derivative neuron) (err neuron)))
    ;; Adjust the weights of the outgoing connections
    (loop
       for cx-node = (head (cx-dlist neuron)) then (next cx-node)
       for cx = (value cx-node)
       do (backpropagate cx)))

  (:method ((cx t-cx))
    ;; Adjust the weight of this connection
    (let* ((delta (* (learning-rate cx)
                     (err-derivative (target cx))
                     (output (source cx))))
           (new-weight (limit-magnitude-and-precision
                        (+ (weight cx) 
                           (+ delta (* (momentum cx) (delta cx)))))))
      (setf (weight cx) new-weight))))

(defmethod apply-inputs ((net t-net) (input-values list))
  (when (zerop (len (layer-dlist net))) 
    (error "Can't apply inputs to a network with no layers."))
  (loop with layer-dlist = (value (head (layer-dlist net)))
     for neuron-node = (head layer-dlist) then (next neuron-node)
     while neuron-node
     for neuron = (value neuron-node)
     for input-value in input-values
     do (setf (input neuron) input-value)
     counting input-value into count
     finally 
       (when (not (equal count (len layer-dlist)))
         (error "~a (~d) differs from ~a (~d)"
                "input-values list count" count
                "input-layer neuron count" (len layer-dlist)))))

(defmethod apply-expected-outputs ((net t-net) (expected-output-values list))
  (when (zerop (len (layer-dlist net)))
    (error "Can't apply outputs to a network with no layers."))
  (loop with layer-dlist = (value (tail (layer-dlist net)))
     for neuron-node = (head layer-dlist) then (next neuron-node)
     while neuron-node
     for neuron = (value neuron-node)
     for expected-output-value in expected-output-values
     do (setf (expected-output neuron) expected-output-value)
     counting expected-output-value into count
     finally
       (when (not (equal count (len layer-dlist)))
         (error "~a (~d) differs from ~a (~d)"
                "expected-output-values list count" count
                "output-layer neuron count" (len layer-dlist)))))

(defmethod collect-outputs ((net t-net))
  (loop with output-layer = (tail (layer-dlist net))
     for neuron-node = (head output-layer) then (next neuron-node)
     while neuron-node collect (output (value neuron-node))))

(defmethod infer-frame ((net t-net) (inputs list))
  (apply-inputs net inputs)
  (feedforward net)
  (collect-outputs net))

(defmethod train-frame ((net t-net) (inputs list) (expected-outputs list))
  (infer-frame net inputs)
  (apply-expected-outputs net expected-outputs)
  (backpropagate net))
  
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
                   :biased (and add-bias (= a n))
                   :transfer-function (getf transfer :function)
                   :transfer-derivative (getf transfer :derivative))
     do (push-tail layer neuron)
     finally (return layer)))

(defun create-net (topology &key (id (bianet-id)) log-file)
  (loop with log = (or log-file (format nil "/tmp/~(~a~).log" id))
     with net = (make-instance 't-net :id id :log-file log)
     for layer-spec in topology
     for count = (or (getf layer-spec :neurons)
                     (error ":neuron-count parameter required"))
     for add-bias = (getf layer-spec :add-bias)
     for transfer-key = (getf layer-spec :transfer-key :logistic)
     for layer = (create-layer count 
                               :add-bias add-bias 
                               :transfer-key transfer-key)
     do (push-tail (layer-dlist net) layer)
     finally (return net)))

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

;; todo
;; (defmethod name-neuron ((neuron t-neuron) (index-in-layer integer))
;;   (setf (name neuron) 
;;         (make-neuron-name (layer neuron) index-in-layer (biased neuron))))

;; (defun make-neuron-name (layer index-in-layer biased)
;;   (format nil "~3,'0d-~9,'0d~a" layer index-in-layer (if biased "b" "a")))

;; todo
;; (defmethod compute-layer-type ((neuron t-neuron) (last-layer integer))
;;   (setf (layer-type neuron)
;;         (cond ((zerop (layer neuron)) :input)
;;               ((= (layer neuron) last-layer) :output)
;;               (t :hidden))))

;; (defmethod find-last-node-in-layer ((net t-net) (layer integer))
;;   (loop for node = (tail (neurons-dlist net)) then (prev node)
;;      while node
;;      for neuron = (value node)
;;      when (equal (layer neuron) layer) do (return node)))

;; (defmethod add-neuron ((net t-net) (neuron t-neuron))
;;   (let ((last-node (find-last-node-in-layer (layer neuron))))
;;     (insert-after-node (neurons-dlist net) last-node neuron)
;;     (index-neurons net)))

;; (defgeneric find-neuron-node (net search)
;;   (:method ((net t-net) (search keyword))
;;     (find-neuron net (lambda (x) (equal (id x) search))))
;;   (:method ((net t-net) (search string))
;;     (find-neuron net (lambda (x) (equal (name x) search))))
;;   (:method ((net t-net) (search function))
;;     (loop for node = (head (neurons-dlist net)) then (next node)
;;        while node
;;        for neuron = (value node)
;;        when (funcall search neuron) do (return node))))

;; (defun find-neuron (net search)
;;   (value (find-neuron-node net search)))

;; (defgeneric drop-neuron (net neuron)
;;   (:method ((net t-net) (neuron t-neuron))
;;     (let ((node (find-neuron-node net (id neuron))))
;;       (when (and node (disconnect-neuron net (value node)))
;;         (delete-node (neurons-dlist net) node)))))

;; (defmethod connectedp ((source t-neuron) (target t-neuron))
;;   (loop with id = (id target)
;;      for cx in (cxs source)
;;      thereis (equal (id (target cx)) id)))

;; (defmethod disconnect-neuron ((net t-net) (neuron t-neuron))
;;   (when (contains-neuron net neuron)
;;     (loop with id = (id neuron)
;;        for node = (head (neurons-dlist net)) then (next node)
;;        while node
;;        for target-neuron = (value node)
         


;; (defmethod disconnect-neuron ((net t-net) (neuron t-neuron))
;;   (when (contains-neuron net neuron)
;;     (loop with id = (id neuron)
;;        for upstream-neuron in (neurons-dlist net)
;;        when (connectedp upstream-neuron neuron)
;;        do (setf (cxs upstream-neuron)
;;                 (loop for cx in (cxs upstream-neuron)
;;                    unless (equal (id (target cx)) id)
;;                    collect cx)))
;;     (loop for cx in (cxs neuron)
;;        do (decf (receptor-count (target cx))))
;;     neuron))       

;; (defun make-layer-neurons (count layer transfer-tag)
;;   (loop with transfer = (gethash transfer-tag *transfers*)
;;      for a from 1 to count collect
;;        (make-instance 't-neuron :layer layer :transfer transfer)))

;; (defmethod neurons-in-layer ((net t-net) (layer-index integer))
;;   (remove-if-not (lambda (n) (equal (layer neuron) layer-index))
;;                  (neurons-dlist net)))

;; (defmethod find-last-layer ((net t-net))
;;   (loop for neuron in (to-list (neurons-dlist net))
;;        maximizing (layer neuron)))

