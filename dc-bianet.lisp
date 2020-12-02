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
                              
(defclass t-transfer ()
  ((name :reader name :initarg :name
         :initform (error ":name required"))
   (function :reader function :initarg :function
             :initform (error ":function required"))
   (derivative :reader derivative :initarg :derivative 
               :initform (error ":derivative required"))))

(defparameter *transfers* (make-hash-table))

(loop with transfer-details = 
     (list (list :logistic #'logistic #'logistic-derivative)
           (list :relu #'relu #'relu-derivative)
           (list :relu-leaky #'relu-leaky #'relu-leaky-derivative))
   for (name function derivative) in transfer-details
   do (setf (gethash name *transfers*)
            (make-instance 't-transfer
                           :name name
                           :function function
                           :derivative derivative)))

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
   (transfer :accessor transfer :initarg :transfer :type t-transfer
             :initform (gethash :logistic *transfers*))
   (output :accessor output :type real :initform 0.0)
   (err :accessor err :type real :initform 0.0)
   (x-coor :accessor x-coor :type real :initform 0.0)
   (y-coor :accessor y-coor :type real :initform 0.0)
   (z-coor :accessor z-coor :type real :initform 0.0)
   (receptor-count :accessor receptor-count :type integer :initform 0)
   (receipts :accessor receipts :type integer :initform 0)
   (cxs :accessor cxs :type list :initform nil)))

(defmethod initialize-instance :after ((neuron t-neuron) &key)
  (when (biased neuron)
    (setf (input neuron) 1.0)))

(defmethod fire ((neuron t-neuron))
  (input-to-output neuron)
  (loop with output = (output neuron)
       for cx in (cxs neuron)
       do (receive (target cx) (* (weight cx) output))))

(defmethod input-to-output ((neuron t-neuron))
  (setf (output neuron)
        (funcall (function (transfer neuron)) (input neuron))))

(defmethod receive ((neuron t-neuron) (value real))
  (incf (input neuron) value)
  (incf (receipts neuron))
  (when (= (receipts neuron) (receptor-count neuron))
    (setf (receipts neuron) 0)
    (fire neuron)))

(defclass t-net ()
  ((id :reader id :initarg :id :type :keyword :initform (bianet-id))
   (topology :accessor topology :initarg :topology :type list :initform nil)
   (neurons :accessor neurons :type list :initform nil)
   (layer-count :accessor layer-count :type integer :initform 0)
   (output-layer :accessor output-layer :type list)
   (input-layer :accessor input-layer :type list)
   (log-file :accessor log-file :type string)
   (stop-training :accessor stop-training :type boolean :initform nil)
   (random-state :accessor random-state :initform (make-random-state))))

(defmethod add-neurons ((net t-net) (neurons list))
  (setf (neurons net)
        (append (neurons net) neurons))
  (index-neurons net))

(defun make-neurons (count layer transfer-tag)
  (loop with transfer = (gethash transfer-tag *transfers*)
     for a from 1 to count collect
       (make-instance 't-neuron :layer layer :transfer transfer)))

(defmethod neurons-in-layer ((net t-net) (layer-index integer))
  (remove-if-not (lambda (n) (equal (layer neuron) layer-index))
                 (neurons net)))

(defmethod find-last-layer ((net t-net))
    (loop for neuron in (neurons net) maximizing (layer neuron)))

(defmethod index-neurons ((net t-net))
  (loop with last-layer = (find-last-layer net)
     with layers = (loop for layer from 0 to last-layer collect nil)
     for neuron in (neurons net)
     do 
       (setf (name neuron) (format nil "~d-~d~a" 
                                   (layer neuron)
                                   ;; The length of 
                                   (length (elt layers (layer neuron)))
                                   (if (biased neuron) "b" ""))
             (layer-type neuron) (cond ((zerop (layer neuron)) :input)
                                       ((= (layer neuron) last-layer) :output)
                                       (t :hidden)))
       (push neuron (elt layers (layer neuron)))
     finally (setf (topology net) (mapcar #'length layers)
                   (input-layer net) (elt layers 0)
                   (output-layer net) (elt layers last-layer))))       
