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
