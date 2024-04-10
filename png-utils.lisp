(defun ensure-directory-string (directory)
  (let ((directory-string (format nil "~a" directory)))
    (if (scan "/$" directory-string)
        directory-string
        (format nil "~a/" directory-string))))

(defun path-tail (directory)
  (car 
   (reverse 
    (remove-if 
     (lambda (s) (zerop (length s)))
     (split "/" (format nil "~a" directory))))))

(defun clear-directory (directory)
  (uiop:delete-directory-tree (pathname directory) :validate t)
  (ensure-directories-exist (ensure-directory-string directory)))

(defun separate-test-files (&key
                              (root-directory
                               "/home/macnod/data/mnist-images")
                              (train-subdirectory "train")
                              (test-subdirectory "test")
                              (test-fraction .05))
  (loop
    with source-directory = (ensure-directory-string
                             (join-paths root-directory train-subdirectory))
    with target-directory = (ensure-directory-string
                             (join-paths root-directory test-subdirectory))
    initially (clear-directory target-directory)
    for source-label-directory in (uiop:subdirectories source-directory)
    for label = (path-tail source-label-directory)
    for source-files = (uiop:directory-files source-label-directory "*.png")
    for target-label-directory = (join-paths target-directory label)
    for target-count = (floor (* (length source-files) test-fraction))
    for source-files-subset = (subseq (shuffle source-files) 0 target-count)
    do (loop for source in source-files-subset
                  for target = (join-paths target-label-directory
                                           (file-namestring source))
                  do (ensure-directories-exist target)
                     (rename-file source target))
    collect (list :label label :sample-size target-count)))


(defun restore-train-files (&key
                            (root-directory
                             "/home/macnod/data/mnist-images")
                            (train-subdirectory "train")
                            (test-subdirectory "test"))
  (loop
    with source-directory = (ensure-directory-string
                             (join-paths root-directory test-subdirectory))
    with target-directory = (ensure-directory-string
                             (join-paths root-directory train-subdirectory))
    for source-label-directory in (uiop:subdirectories source-directory)
    for source-files = (uiop:directory-files source-label-directory "*.png")
    for target-label-directory = (join-paths
                                  target-directory
                                  (path-tail source-label-directory))
    do (loop for source in source-files
             for target = (join-paths target-label-directory
                                      (file-namestring source))
             do (rename-file source target))))

(defun directory-tree-to-csv (&key
                              (root-directory
                               "/home/macnod/data/mnist-images")
                              (source-subdirectory "test")
                              (target-csv-file 
                               (format nil "~a.csv" source-subdirectory)))
  (with-open-file (csv (join-paths root-directory target-csv-file)
                       :direction :output
                       :if-does-not-exist :create
                       :if-exists :supersede)
    (loop
      with directories = (uiop:subdirectories 
                          (ensure-directory-string
                           (join-paths root-directory source-subdirectory)))
      for label in (mapcar #'path-tail directories)
      for directory in directories
      for directory-files = (uiop:directory-files directory)
      do (loop for file in directory-files
               for inputs = (normalize-list (read-png file))
               do (format csv "~s,~{~a~^,~}~%" label inputs)))))
    
(defun infer-directory-pngs (environment-id
                             label
                             &key (root-directory
                                   "/home/macnod/data/mnist-images/test"))
  (loop for file in (uiop:directory-files 
                     (format nil "~a/" (join-paths root-directory label)))
        collect (list :file (file-namestring file)
                      :label (infer-png-file environment-id file))
          into inferred-labels
        finally (return (list :actual-label label
                              :inferred-labels inferred-labels))))
