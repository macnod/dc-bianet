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
            :initform #'limit-magnitude-and-precision)))

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
   (layer :accessor layer :initarg :layer :type integer 
          :initform 1) ;; layer 0 is the input layer
   (layer-type :accessor layer-type :initarg :layer-type)
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
   (err :accessor err :type real :initform 0.0)
   (x-coor :accessor x-coor :type real :initform 0.0)
   (y-coor :accessor y-coor :type real :initform 0.0)
   (z-coor :accessor z-coor :type real :initform 0.0)
   (receptor-count :accessor receptor-count :type integer :initform 0)
   (receipt-count :accessor receipt-count :type integer :initform 0)
   (effectors :accessor effectors :type list :initform nil)
   (receptors :accessor receptors :type list :initform nil)))

(defmethod initialize-instance :after ((neuron t-neuron) &key)
  (when (biased neuron)
    (setf (input neuron) 1.0)))

(defgeneric fire (thing)
  (:method ((net t-net))
    (loop with dlist = (neurons net)
       for node = (head dlist) then (next node)
       while node
       for neuron = (value node)
       while (zerop (layer neuron))
       do (fire neuron)))
  (:method ((neuron t-neuron))
    (loop with output = (input-to-output neuron)
       for cx in (effectors neuron) do (fire cx)))
  (:method ((cx t-cx))
    (let ((target (target cx))
          (source (source cx))
          (weight (weight cx)))
      (incf (input target) (* weight (output source)))
      (incf (receipt-count target))
      (when (= (receipt-count target) (receptor-count target))
        (fire target)))))

(defmethod input-to-output ((neuron t-neuron))
  (let ((input (input neuron))
        (output (funcall (transfer-function neuron) input)))
    (setf (input-shadow neuron) input
          (input neuron) (if biased input 0.0)
          (receipt-count neuron) 0
          (output neuron) output
          (err neuron) nil)))

(defclass t-net ()
  ((id :reader id :type :keyword :initform (bianet-id))
   (name :accessor name :type string)
   (topology :accessor topology :initarg :topology :type list :initform nil)
   (neurons :accessor neurons :type dlist :initform (make-instance 'dlist))
   (layer-count :accessor layer-count :type integer :initform 0)
   (first-output-node :accessor first-output-node :type dlist-node)
   (log-file :accessor log-file :type string)
   (stop-training :accessor stop-training :type boolean :initform nil)
   (random-state :accessor random-state :initform (make-random-state))))

(defmethod index-neurons ((net t-net))
  (loop with last-layer = (find-last-layer net)
     with layers = (loop for layer from 0 to last-layer collect nil)
     for neuron in (neurons net)
     for index-in-layer = (length (elt layers (layer neuron)))
     do 
       (name-neuron neuron index-in-layer)
       (compute-layer-type neuron last-layer)
       (push neuron (elt layers (layer neuron)))
     finally 
       (setf (neurons net) (sorted (neurons net)
                                   (lambda (a b) (string< (name a) (name b))))
             (topology net) (mapcar #'length layers)
             (first-output net) (car (elt layers last-layer)))))

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
  (loop for node = (tail (neurons net)) then (prev node)
     while node
     for neuron = (value node)
     when (equal (layer neuron) layer) do (return node)))

(defmethod add-neuron ((net t-net) (neuron t-neuron))
  (let ((last-node (find-last-node-in-layer (layer neuron))))
    (insert-after-node (neurons net) last-node neuron)
    (index-neurons net)))

(defgeneric find-neuron-node (net search)
  (:method ((net t-net) (search keyword))
    (find-neuron net (lambda (x) (equal (id x) search))))
  (:method ((net t-net) (search string))
    (find-neuron net (lambda (x) (equal (name x) search))))
  (:method ((net t-net) (search function))
    (loop for node = (head (neurons net)) then (next node)
       while node
       for neuron = (value node)
       when (funcall search neuron) do (return node))))

(defun find-neuron (net search)
  (value (find-neuron-node net search)))

(defgeneric drop-neuron (net neuron)
  (:method ((net t-net) (neuron t-neuron))
    (let ((node (find-neuron-node net (id neuron))))
      (when (and node (disconnect-neuron net (value node)))
        (delete-node (neurons net) node)))))

(defmethod connectedp ((source t-neuron) (target t-neuron))
  (loop with id = (id target)
     for cx in (cxs source)
     thereis (equal (id (target cx)) id)))

(defmethod disconnect-neuron ((net t-net) (neuron t-neuron))
  (when (contains-neuron net neuron)
    (loop with id = (id neuron)
       for node = (head (neurons net)) then (next node)
       while node
       for target-neuron = (value node)
         


(defmethod disconnect-neuron ((net t-net) (neuron t-neuron))
  (when (contains-neuron net neuron)
    (loop with id = (id neuron)
       for upstream-neuron in (neurons net)
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
                 (neurons net)))

(defmethod find-last-layer ((net t-net))
    (loop for neuron in (neurons net) maximizing (layer neuron)))

