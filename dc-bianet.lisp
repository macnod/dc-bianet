(in-package :dc-bianet)

(defparameter *magnitude-limit* 1e9)
(defparameter *precision-limit* 1e-9)
          
(defun limit-magnitude-and-precision (x)
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

(defclass t-cx ()
  ((source :reader source :initarg :source :type t-neuron
           :initform (error ":source required"))
   (target :reader target :initarg :target :type t-neuron
           :initform (error ":target required"))
   (weight :accessor weight :initarg :weight :initform 0.1 :type real)
   (learning-rate :accessor learning-rate :initarg :learning-rate :type real
                  :initform 0.1)
   (momentum :accessor momentum :initarg :momentum :initform 0.1)
   (last-delta :accessor delta :initarg :delta :initform 0.0)
   (limiter :accessor limiter :initarg :limiter
            :initform #'limit-magnitude-and-precision)
   (node :accessor node :initarg :node :initform nil)))

(defmethod initialize-instance :after ((cx t-cx) &key)
  (push cx (effectors (source cx)))
  (push cx (receptors (target cx)))
  (incf (receptor-count (target cx))))

(defmethod adjust-weight ((cx t-cx))
  (let* ((delta (* (learning-rate cx)
                   (compute-neuron-error (target cx))
                   (output (source cx))))
         (new-weight (+ (weight cx)
                        (+ delta (* (momentum cx) (last-delta cx))))))
    (setf (weight cx)
          (funcall (limiter cx) new-weight))))

(defclass t-neuron ()
  ((id :accessor id :initarg :id :type keyword :initform (bianet-id))
   (name :accessor name :initarg :id :type string)
   ;; Valid values: :input, :hidden, :output
   (layer-type :accessor layer-type :initarg :layer-type :initform :hidden)
   (biased :access biased :initarg :biased :type boolean :initform nil)
   (input :accessor input :type real :initform 0.0)
   (input-shadow :accessor input-shadow :type real :initform 0.0)
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
   (err-derivative :accessor :err-derivative :type real :initform 0.0)
   (x-coor :accessor x-coor :type real :initform 0.0)
   (y-coor :accessor y-coor :type real :initform 0.0)
   (z-coor :accessor z-coor :type real :initform 0.0)
   (cx-dlist :accessor cx-dlist :type dlist :initform (make-instance 'dlist))
   (node :accessor node :type dlist-node :initform nil)))

(defmethod initialize-instance :after ((neuron t-neuron) &key)
  (when (biased neuron)
    (setf (input neuron) 1.0)))

(defmethod input-to-output ((neuron t-neuron))
  (let ((input (input neuron))
        (output (funcall (transfer-function neuron) input)))
    (setf (output neuron) output
          (input neuron) (if biased input 0.0)
          (input-shadow neuron) input
          (err neuron) nil)))

(defmethod output-to-input ((neuron t-neuron))
  (setf (err-derivative neuron)
        (funcall (transfer-derivative neuron) (err neuron))))

(defmethod integegrate ((neuron t-neuron) (value real))
  (incf (input neuron) value)
  (incf (receipts neuron)))

(defclass t-net ()
  ((id :reader id :type :keyword :initform (bianet-id))
   (name :accessor name :type string)
   (layers-dlist :accessor layers-dlist :type dlist :initform (make-instance 'dlist))
   (log-file :accessor log-file :type string)
   (stop-training :accessor stop-training :type boolean :initform nil)
   (random-state :accessor random-state :initform (make-random-state))))

(defgeneric feedforward (thing)

  (:method ((net t-net))
    (loop 
       for layer-node = (head (layers-dlist net)) then (next layer-node)
       while layer-node
       do (feedforward (value layer-node))))

  (:method ((layer dlist))
    (loop 
       for neuron-node = (head layer) then (next neuron-node)
       while neuron-node
       do (feedforward (value neuron-node))))

  (:method ((neuron t-neuron))
    (loop initially (input-to-output neuron)
       for cx-node = (head (cx-dlist neuron)) then (next cx-node)
       while cx-node
       do (feedforward (value cx-node))))

  (:method ((cx t-cx))
    (integrate (target cx) (* (weight cx) (output (source cx))))))

(defmethod backpropagate (thing)
  
  (:method ((net t-net))
    (loop
       for layer-node = (tail (layers-dlist net)) then (prev layer-node)
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
              (- (expected-output neuron) (output neuron))
              (loop
                 for cx-node = (head (cx-dlist neuron)) then (next cx-node)
                 while cx-node
                 sum (backpropagate (value cx-node)))))
    (output-to-input neuron))

  (:method ((cx t-cx))
    (* (weight cx) (err (target cx)))))

(defmethod apply-inputs ((net t-net) (input-values list))
  (when (zerop (len (layers-dlist net))) 
    (error "Can't apply inputs to a network with no layers."))
  (loop with layer-dlist = (value (head (layers-dlist net)))
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
  (when (zerop (len (layers-dlist net)))
    (error "Can't apply outputs to a network with no layers."))
  (loop with layer-dlist = (value (tail (layers-dlist net)))
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

(defmethod wave ((net t-net) (inputs list) (expected-outputs list))
  (apply-inputs net inputs)
  (feed-forward net)
  (if expected-outputs
      (progn
        (apply-expected-outputs expected-outputs)
        (backpropagate net)
        (loop with output-layer-dlist = (tail (layers-dlist net))
           for neuron-node = (head output-layer-dlist) then (next neuron-node)
           while neuron-node
           for neuron = (value neuron-node)
           summing (expt (err neuron) 2) into network-error))
      (loop with output-layer-dlist = (tail (layer-dlist net))
         for neuron-node = (head output-layer-dlist) then (next neuron-node)
         while neuron-node
         for neuron = (value neuron-node)
         collect (output neuron))))

(defmethod name-neuron ((neuron t-neuron) (index-in-layer integer))
  (setf (name neuron) 
        (make-neuron-name (layer neuron) index-in-layer (biased neuron))))

(defun make-neuron-name (layer index-in-layer biased)
  (format nil "~3,'0d-~9,'0d~a" layer index-in-layer (if biased "b" "a")))

(defmethod compute-layer-type ((neuron t-neuron) (last-layer integer))
  (setf (layer-type neuron)
        (cond ((zerop (layer neuron)) :input)
              ((= (layer neuron) last-layer) :output)
              (t :hidden))))

(defmethod find-last-node-in-layer ((net t-net) (layer integer))
  (loop for node = (tail (neurons-dlist net)) then (prev node)
     while node
     for neuron = (value node)
     when (equal (layer neuron) layer) do (return node)))

(defmethod add-neuron ((net t-net) (neuron t-neuron))
  (let ((last-node (find-last-node-in-layer (layer neuron))))
    (insert-after-node (neurons-dlist net) last-node neuron)
    (index-neurons net)))

(defgeneric find-neuron-node (net search)
  (:method ((net t-net) (search keyword))
    (find-neuron net (lambda (x) (equal (id x) search))))
  (:method ((net t-net) (search string))
    (find-neuron net (lambda (x) (equal (name x) search))))
  (:method ((net t-net) (search function))
    (loop for node = (head (neurons-dlist net)) then (next node)
       while node
       for neuron = (value node)
       when (funcall search neuron) do (return node))))

(defun find-neuron (net search)
  (value (find-neuron-node net search)))

(defgeneric drop-neuron (net neuron)
  (:method ((net t-net) (neuron t-neuron))
    (let ((node (find-neuron-node net (id neuron))))
      (when (and node (disconnect-neuron net (value node)))
        (delete-node (neurons-dlist net) node)))))

(defmethod connectedp ((source t-neuron) (target t-neuron))
  (loop with id = (id target)
     for cx in (cxs source)
     thereis (equal (id (target cx)) id)))

(defmethod disconnect-neuron ((net t-net) (neuron t-neuron))
  (when (contains-neuron net neuron)
    (loop with id = (id neuron)
       for node = (head (neurons-dlist net)) then (next node)
       while node
       for target-neuron = (value node)
         


(defmethod disconnect-neuron ((net t-net) (neuron t-neuron))
  (when (contains-neuron net neuron)
    (loop with id = (id neuron)
       for upstream-neuron in (neurons-dlist net)
       when (connectedp upstream-neuron neuron)
       do (setf (cxs upstream-neuron)
                (loop for cx in (cxs upstream-neuron)
                   unless (equal (id (target cx)) id)
                   collect cx)))
    (loop for cx in (cxs neuron)
       do (decf (receptor-count (target cx))))
    neuron))       

(defun make-layer-neurons (count layer transfer-tag)
  (loop with transfer = (gethash transfer-tag *transfers*)
     for a from 1 to count collect
       (make-instance 't-neuron :layer layer :transfer transfer)))

(defmethod neurons-in-layer ((net t-net) (layer-index integer))
  (remove-if-not (lambda (n) (equal (layer neuron) layer-index))
                 (neurons-dlist net)))

(defmethod find-last-layer ((net t-net))
  (loop for neuron in (to-list (neurons-dlist net))
       maximizing (layer neuron)))

