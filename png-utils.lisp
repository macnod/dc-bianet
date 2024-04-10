(defun separate-test-files (&key
                              (root-directory
                               "/home/macnod/data/mnist-images")
                              (subdirectories
                               (loop for x from 0 to 9
                                     collect (format nil "~a" x)))
                              (test-fraction .05))
  (loop
    for directory in subdirectories
    for source-directory = (format nil "~a/"
                                   (join-paths root-directory "train" directory))
    for source-files = (uiop:directory-files source-directory "*.png")
    for target-directory = (join-paths root-directory "test" directory)
    for target-count = (floor (* (length source-files) test-fraction))
    for source-files-subset = (subseq (shuffle source-files) 0 target-count)
    do (loop for source in source-files-subset
             for target = (join-paths target-directory
                                      (file-namestring source))
             do (ensure-directories-exist target)
                (rename-file source target))))

;; (defun directory-tree-to-csv (&key
;;                               (root-directory
;;                                "/home/macnod/data/mnist-images/train"))
;;   (loop
;;     with labels = (mapcar
;;                    (lambda (s) (car (reverse (split "/" (format nil "~a" s)))))
;;                    (uiop:subdirectories root-directory))
;;     for label in labels
;;     for label-files 
