(in-package :dc-bianet)

(defun infer-directory-pngs (environment-id label-directory)
  (infer-label-directory-files
   label-directory
   (lambda (file)
     (infer-png environment-id file))))
   

;; Needs work. Should pick the largest image in the tree. This will also require
;; changing the neural network so that it can correctly process variable-length
;; vectors.
(defun png-tree->suggest-topology (png-tree-path)
  (loop with output-count = (length (subdirectory-names png-tree-path))
     with input-count = (length (read-png (example-file png-tree-path ".png")))
     for power = 1 then (1+ power)
     while (< (expt 2 power) output-count)
     finally (return (list input-count (expt 2 (1+ power)) output-count))))

(defun png-tree->frames (directory &key as-vector)
  (loop with labels = (subdirectory-names directory)
     with label->index = (list->key-index labels)
     with label->expected-outputs = (label-outputs-hash label->index)
     for label in labels
     for label-folder = (join-paths directory label)
     for expected-outputs = (gethash label label->expected-outputs)
     appending (pngs->frames-for-label label-folder expected-outputs)
     into frames
     finally (return (if as-vector
                         (map 'vector 'identity frames)
                         frames))))

(defun png-data-set-to-csv (data-set-directory)
  (data-set-directory-to-csv
   data-set-directory
   (lambda (file) (normalize-list (read-png file)))))

(defun infer-png (environment-id file)
  (let ((environment (environment-by-id environment-id)))
    (outputs-label
     (label-vector (output-labels environment))
     (infer (net environment)
            (normalize-list (read-png file) :min 0 :max 255)))))

(defun infer-pngs (environment-id label-directory)
  (infer-label-directory-files
   label-directory
   (lambda (file) (infer-png environment-id file))))

(defun read-png (filename &key (width 28) (height 28))
  (loop with image-data = (png-read:image-data
                           (png-read:read-png-file filename))
        with dimensions = (length (array-dimensions image-data))
        for y from 0 below height 
        appending (loop for x from 0 below width 
                        collecting (if (= dimensions 2)
                                       (aref image-data x y)
                                       (aref image-data x y 0)))
          into intensity-list
        finally (return (invert-intensity intensity-list))))

(defun invert-intensity (list &key (max 255))
  (loop for element in list collect (- max element)))

(defun png-to-inputs (filename width height)
  (normalize-list
   (read-png filename :width width :height height) :min 0 :max 255))

;; (defun inputs->png (inputs filename)
;;   (loop with png = (make-instance 'png
;;                                   :color-type :grayscale
;;                                   :width 28
;;                                   :height 28)
;;      with image = (data-array png)
;;      for input in inputs
;;      for x = 0 then (mod (1+ x) 28)
;;      for y = 0 then (if (zerop x) (1+ y) y)
;;      do (setf (aref image y x 0) (- 255 (truncate (* input 255))))
;;      finally (zpng:write-png png filename)))

;; (defun training-set->pngs (id path &key label count)
;;   (loop with environment = (environment-by-id id)
;;      for (frame-label inputs) in
;;        (loop with frame-count = 0
;;           for frame across (training-set environment)
;;           for frame-label = (outputs->label environment (second frame))
;;           for frame-inputs = (car frame)
;;           for is-match = (or (null label) (equal frame-label label))
;;           when is-match collect (list frame-label frame-inputs) into input-lists
;;           and do (incf frame-count)
;;           when (and count (>= frame-count count)) do (return input-lists)
;;           finally (return input-lists))
;;      for index = 1 then (1+ index)
;;      for dir = (join-paths path frame-label)
;;      for filename = (join-paths dir (format nil "~3,'0d.png" index))
;;      do (ensure-directories-exist (concatenate 'string dir "/"))
;;        (inputs->png inputs filename)))

;; (defun classify-pngs (environment path prefix)
;;   (declare (ignore environment))
;;   (loop with dir-spec = (format nil "~a-*.png" (join-paths path prefix))
;;      for file in (directory dir-spec)
;;      for normalized-file-data = (normalize-list (read-png file) :min 0 :max 255)
;;      for outputs = (infer *net* normalized-file-data)
;;      for label = (outputs->label *environment* outputs)
;;      collect (list (file-namestring file) label)))

;; (defun train-on-png (id label file count)
;;   (loop with environment = (environment-by-id id)
;;      with net = (net environment)
;;      with frame = (png-file->frame :digits label file)
;;      with inputs = (car frame)
;;      with expected-outputs = (second frame)
;;      for a from 1 to count do (train-frame net frame)
;;      finally (return (outputs->label environment (infer net inputs)))))

;; (defun pngs->frames-for-label (label-folder expected-outputs &key as-vector)
;;   (loop
;;     with dir-spec = (format nil "~a/*.png" label-folder)
;;     for file in (directory dir-spec)
;;     for inputs = (normalize-list (read-png file) :min 0 :max 255)
;;     collect (list inputs expected-outputs) into frames
;;     finally (return (if as-vector
;;                         (map 'vector 'identity frames)
;;                         frames))))
