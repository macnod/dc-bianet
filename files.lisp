;; This code deals with files that exist in a project tree like this
;; one:
;;
;;   PROJECT            <-- project directory
;;    ├─train           <-- data-set directory
;;    │ ├─cat           <-- label directory
;;    │ │ ├─0001.png    <-- example file
;;    │ │ ├─0002.png
;;    │ │ └─0003.png
;;    │ │   ⋮
;;    │ └─dog           <-- label directory
;;    │   ├─1001.png
;;    │   ├─1002.png
;;    │   └─1003.png
;;    │     ⋮
;;    └─test            <-- data-set directory
;;      ├─cat           <-- label directory
;;      │ ├─2001.png
;;      │ ├─2002.png    <-- example file
;;      │ └─2003.png
;;      │   ⋮
;;      └─dog           <-- label directory
;;        ├─3001.png
;;        ├─3002.png
;;        └─3003.png    <-- example file
;;          ⋮
;;
;; In the directory tree depicted above, "PROJECT" is the project
;; directory. The project directory can contain one or more data
;; sets, depicted by the "train" and "test" directories above.
;; A data set contains subdirectories that represent labels,
;; with each label subdirectory containing files with examples
;; associated with the label. The files be in any format, as
;; long as the format is consistent for all the files in the
;; project directory.
;;
;;   - project directory: Top-level directory for a project.  This
;;     directory should contain data-set directories.
;;
;;   - data-set directory: A directory containing label
;;     directories. It should be a child of a project directory
;;
;;   - label directory: A directory that represents a label, and that
;;     contains example files that are representative of that label.
;;
;;   - example file: A file that contains data that is an example
;;     of the label represented by its parent directory.
;;
;; This code does not convert files into vectors that are
;; suitable for dc-bianet training. Instead, this code provides
;; more general functions for dealing with the filesa and the
;; tree.

(in-package :dc-bianet)

(defun stringify (x)
  "When we're dealing with building paths, we might want numbers and
file paths to behave like strings."
  (when x (format nil "~a" x)))

(defun get-labels (path &key (sorted t))
  "Returns a list of labels from PATH. If SORTED is T, the labels are
 sorted alphabetically. Otherwise, the labels are returned in the
 order they are found. SORTED is T by default.

If PATH is a directory, assume that the directory is a data-set
directory, and create a list of labels by collecting the names of the
subdirectories of PATH and returning them as a list of string.

If PATH is a file, assume that the file is CSV file where the first
element of each row is the label, and return a list of the unique
labels in the file."
  (let ((list (case (path-type path)
                (:file (csv-sample-labels path))
                (:directory (subdirectory-names path))
                (:not-found (error "File not found: ~a" path)))))
    (if sorted (sort list #'string<) list)))

(defun subdirectory-names (directory)
  "This function returns a list of the names of the subdirectories of the
directory given by PATH. These names are plain strings with no
directory slashes. This is useful for obtaining the labels associated
with a data-set directory."
  (when (not (directory-exists-p directory))
    (error "Not a directory or directory doesn't exist: ~a" directory))
  (mapcar
   (lambda (d) (car (last (pathname-directory d))))
   (uiop:subdirectories (ensure-directory-string directory))))

(defun csv-sample-labels (csv-file)
  "Picks the first element of each row in the file at CSV-FILE, forcing
the element to a string, and returns a distinct list of the elements.

For performance reasons, this function does not support the CSV format
completely. That first element may not contain commas, even inside of
quotes."
  (t:transduce
   (t:comp
    (t:filter (lambda (s) (not (zerop (length s)))))
    (t:map (lambda (s)
             (string-trim 
              "\""
              (format nil "~a"
                      (read-from-string
                       (subseq s 0 (position #\, s)))))))
    #'t:unique)
   #'t:cons
   (pathname csv-file)))

(defun data-set-file-counts (directory)
  "Returns a single integer representing the number of files that exist
in directories that are direct children of DIRECTORY."
  (loop
    for name in (subdirectory-names directory)
    summing (length 
             (uiop:directory-files 
              (ensure-directory-string 
               (join-paths directory name))))))

(defun data-set-files (directory)
  "Returns a hash table that maps each label to a list of the absolute
file names in the label's directory under DIRECTORY. This function
assumes that DIRECTORY is a data-set directory, and that the names of
the subdirectories of DIRECTORY correspond to the data-set labels.
See the comments at the beginning of this module for more a
description of a data-set directory."
  (loop
    with label-files = (make-hash-table :test 'equal)
    for label in (subdirectory-names directory)
    do (setf (gethash label label-files)
             (uiop:directory-files 
              (ensure-directory-string (join-paths directory label))))
    finally (return label-files)))

(defun ensure-directory-string (directory)
  (let ((directory-string (format nil "~a" directory)))
    (if (scan "/$" directory-string)
        directory-string
        (format nil "~a/" directory-string))))

(defun path-tail (directory)
  "Returns the last element of DIRECTORY, as a plain string without
slashes. For example:

    (path-tail \"/data/cats-and-dogs/train/cat\") -> \"cat\"

This can be useful when determining the label that a directory
represents."
  (car
   (last
    (remove-if
     (lambda (s) (zerop (length s)))
     (split "/" (format nil "~a" directory))))))

(defun clear-directory (directory)
  "Completely erases the content of DIRECTORY."
  (when (directory-exists-p directory)
    (uiop:delete-directory-tree (pathname directory) :validate t)
    (ensure-directories-exist (ensure-directory-string directory))))

(defun separate-test-files (project-directory
                            &key
                              (source-subdirectory "train")
                              (target-subdirectory "test")
                              (test-fraction .05))
  "Moves TEST-FRACTION of files from the label directories in
SOURCE-SUBDIRECTORY to the label directories in
TARGET-SUBDIRECTORY. SOURCE-SUBDIRECTORY and TARGET-SUBDIRECTORY are
relative data-set directory names in the form of strings without
directory separators. These data-set directories exist in
PROJECT-DIRECTORY. SOUCE-SUBDIRECTORY contains a number of label
directories. This function empties TARGET-SOURCE-DIRECTORY, creates
the label directories there, and moves some files from the label
directories in SOURCE-SUBDIRECTORY to the new label directories in
TARGET-SUBDIRECTORY.

The function computes the number of files to move
in each source label directory by multiplying TEST-FRACTION by the
number of files in the label directory.

The function picks the specific files to move at random, from among
the files in the label directories.

Calling this function is helpful when you want to set aside some
examples for testing the neural network when training is
complete. Typically, these examples go in the data-set directory
\"test\"."
  (loop
    with source-directory = (ensure-directory-string
                             (join-paths 
                              project-directory source-subdirectory))
    and target-directory = (ensure-directory-string
                            (join-paths project-directory target-subdirectory))
    initially (clear-directory target-directory)
    for source-label-directory in (uiop:subdirectories source-directory)
    for label = (path-tail source-label-directory)
    for source-files = (uiop:directory-files source-label-directory)
    for target-label-directory = (join-paths target-directory label)
    for target-count = (floor (* (length source-files) test-fraction))
    for source-files-subset = (subseq (shuffle source-files) 0 target-count)
    do (loop for source in source-files-subset
                  for target = (join-paths target-label-directory
                                           (file-namestring source))
                  do (ensure-directories-exist target)
                     (rename-file source target))
    collect (list :label label :sample-size target-count)))

(defun restore-train-files (project-directory
                            &key
                            (source-subdirectory "test")
                            (target-subdirectory "train"))
  "This function attempts to undo the work that SEPARATE-TEST-FILES does,
moving files from SOURCE-SUBDIRECTORY (now typically the \"test\"
directory) back to TARGET-SUBDIRECTORY (now typically the \"train\"
directory)."
  (loop
    with source-directory = (ensure-directory-string
                             (join-paths project-directory source-subdirectory))
    and target-directory = (ensure-directory-string
                            (join-paths project-directory target-subdirectory))
    for source-label-directory in (uiop:subdirectories source-directory)
    for source-files = (uiop:directory-files source-label-directory)
    for target-label-directory = (join-paths
                                  target-directory
                                  (path-tail source-label-directory))
    do (loop for source in source-files
             for target = (join-paths target-label-directory
                                      (file-namestring source))
             do (rename-file source target))))

(defun data-set-directory-to-csv (data-set-directory
                                  file-transformation
                                  &key target-csv-file)
  "Creates the csv file TARGET-CSV-FILE. Each row of this csv file
consists of a label followed by numbers. The label comes from the
label directory names and the numbers from from a transformation of
the file into a list of numbers, using the FILE-TRANSFORMATION
function, which accepts a single parameter that consists of the
file's absolute path name.

If you dont specify TARGET-CSV-FILE, then the function computes the
file path by identifying the parent directory of DATA-SET-DIRECTORY
and placing the file there, with the same name as the last directory
in the DATA-SET-DIRECTORY path, but with the extension \".csv\". For
example, for the following DATA-SET-DIRECTORY value:

  /data/dog-or-cat/train/

Not specifying TARGET-CSV-FILE is the same as specifying this value:

  /data/dog-or-cat/train.csv

The resulting csv file should be easier for other neural networks to
process for training.

Aside from creating the csv file, this function returns a report
with the path, size, and line-count of the created file.
"
  (let* ((root (ensure-directory-string data-set-directory))
         (target (or target-csv-file
                     (format nil "~a.csv"
                             (join-paths
                              (uiop:pathname-parent-directory-pathname root)
                              (path-tail root))))))
    (with-open-file (csv target
                         :direction :output
                         :if-does-not-exist :create
                         :if-exists :supersede)
      (loop
        with directories = (uiop:subdirectories root)
        and line-count = 0
        for label in (mapcar #'path-tail directories)
        for directory in directories
        for directory-files = (uiop:directory-files directory)
        do (loop for file in directory-files
                 for inputs = (funcall file-transformation file)
                 do (format csv "~s,~{~a~^,~}~%" label inputs)
                    (incf line-count))
        finally
           (let ((size (file-length csv)))
             (return (list :target target
                           :size size
                           :human-readable (format nil "~:D" size)
                           :line-count line-count)))))))

(defun infer-label-directory-files (label-directory inference-function)
  "Identifies the label and associated files in LABEL-DIRECTORY,
applies INFERENCE-FUNCTION to each file, and returns a report of the
performance and accuracy of the inferences. Use this function to
evaluate the accuracy of a neural network against a training or
testing data set. This function is not for performing inferences on
new, unlabeled data."
  (loop 
    with label = (path-tail label-directory)
    and label-files = (uiop:directory-files
                       (ensure-directory-string label-directory))
    for file in label-files
    for file-count = 1 then (1+ file-count)
    for inferred-label = (funcall inference-function file)
    for correct = (string= label inferred-label)
    for correct-count = (if correct 1 0)
    then (+ correct-count (if correct 1 0))
    when correct do (incf correct-count)
    collect (list :file (file-namestring file)
                  :label (funcall inference-function file)
                  :correct correct)
      into inferred-labels
    finally 
       (return 
         (list :actual-label label
               :accuracy (format nil "~3,f%" (/ correct-count file-count))
               :inferred-labels inferred-labels))))

(defun csv-label-sample-size (csv-file)
  "Picks the first element of each row in the file at CSV-FILE, forcing
the element to a string, and returns a hash table with each element
as a key and the number of times the element was encountered as a the
value.

For performance reasons, this function does not support the CSV format
completely. That first element may not contain commas, even inside of
quotes."
  (hashify-list 
   (t:transduce
    (t:comp
     (t:filter (lambda (s) (not (zerop (length s)))))
     (t:map (lambda (s)
              (string-trim 
               "\""
               (format nil "~a"
                       (read-from-string
                        (subseq s 0 (position #\, s))))))))
     #'t:cons
     (pathname csv-file))))

(defun example-label-file (data-set-directory &optional rstate)
  "Returns a random example file from a random label directory in
 DATA-SET-DIRECTORY. You can use RSTATE to make the function behave
 deterministically. See the documentation for the Common Lisp function
 MAKE-RANDOM-STATE for information about the value of RSTATE."
  (choose-one
   (uiop:directory-files
    (choose-one
     (uiop:subdirectories (ensure-directory-string data-set-directory))
     rstate))
   rstate))

(defun label-sample-size (data-set-directory)
  "Returns a hash table where each key is a label in DATA-SET-DIRECTORY
and each value is the the count of files in that label's directory."
  (loop 
    with h = (make-hash-table :test 'equal)
    for label in (subdirectory-names data-set-directory)
    for samples = (uiop:directory-files
                   (ensure-directory-string
                    (join-paths data-set-directory label)))
    do (setf (gethash label h) (length samples))
    finally (return h)))

(defun normalize-sample-set (sample-set)
  "Normalizes the values in the inputs list of each sample in vector
SAMPLE-SET. For normalization, the lowest and highest input values
across all vector elements is identified first, then in a second pass,
the data is normalized."
  (loop
    with min-max = (loop 
                     for frame across sample-set
                     for inputs = (car frame)
                     for frame-minmax = (loop for input across inputs
                                              minimize input into min
                                              maximize input into max
                                              finally (return (cons min max)))
                     minimize (car frame-minmax) into frame-min
                     maximize (cdr frame-minmax) into frame-max
                     finally (return (cons frame-min frame-max)))
    with range = (float (- (cdr min-max) (car min-max)))
    ;; All vector elements have input lists of the same length,
    ;; so we can use the first element to determine that length.
    and input-count = (length (car (aref sample-set 0)))
    and min-value = (car min-max)
    for frame across sample-set
    for input-values = (car frame)
    do (loop 
         for value in input-values
         collect (/ (- value min-value) range) into normalized
         finally (setf (car frame) normalized))))

(defun create-sample-set (data-set-path 
                          label-outputs
                          file-transformation)
  (case (path-type data-set-path)
    (:file (create-sample-set-from-file data-set-path label-outputs))
    (:directory (create-sample-set-from-directory data-set-path 
                                                  label-outputs 
                                                  file-transformation))
    (:t (error "File not found: ~a" data-set-path))))

(defun split-csv-line (line)
  "This function splits the string LINE into values, sort of like a CSV
parser splits CSV lines into values. It does not handle all CSV lines
correctly or in the same way that an real CSV parser would. For
example, this function doesn't care if the commas are inside of a
quoted value. However, this function is much faster than, a standard
parser, because it does significantly less work."
  (loop 
    for start = 0 then (1+ end)
    for end = (position #\, line :start start)
    while end
    collect (subseq line start end) into values
    finally (return (append values (list (subseq line start))))))

(defun create-sample-set-from-file (data-set-path label-outputs)
  "Reads the file at DATA-SET-PATH and turns it into a neural network
sample set. A sample set is a vector of lists, where each list
contains a list of input values and a list of expected output
values. See the definition of class ALPHA-ENVIRONMENT, in the
dc-bianet module, for more information about the structure of a sample
set. Specifically, look at the documentation of the TRAINING-SET class
attribute.

LABEL-OUTPUTS is a hash table that maps each label string to a list of
floating-point values, which are the expected-output values for the
label. For more information about the LABEL-OUTPUTS hash, see the
documentation for the function LABEL-OUTPUTS, in the labels module.

The LABEL-OUTPUTS hash table could easily be computed from the data in
the file at DATA-SET-PATH, but any caller is already likely to have
the LABEL-OUTPUTS hash table handy already, so this function requires
the parameter to save itself the work of computing the hash table
again.

The file at DATA-SET-PATH should be a CSV-like file, with lines that
contain comma-separated values. The first value in each line should be
a string (optionally quoted), but that string cannot contain a
comma. The rest of the values in each line should be floating-point
values. The number of floating-point values should correspond exactly
with the number of inputs in the neural network that will use this
sample set for training, and with the length of the output lists in
LABEL-OUTPUTS. The count of distinct labels in the file should be
equal to the hash-table count of LABEL-OUTPUTS."
  (loop
    with lines = (t:transduce
                  (t:comp
                   (t:filter (lambda (s) (not (zerop (length s)))))
                   (t:map (lambda (s) (split "," s))))
                  #'t:cons
                  (pathname data-set-path))
    with sample-set = (make-array (length lines)
                                  :element-type 'list
                                  :initial-element nil)
    with input-count = (1- (length (car lines)))
    for line in lines
    for i = 0 then (1+ i)
    for label = (string-trim "\" " (car line))
    for inputs = (mapcar #'read-from-string (subseq line 1 (1+ input-count)))
    for outputs = (gethash label label-outputs)
    do (setf (aref sample-set i) (list inputs outputs))
    finally (return sample-set)))

(defun create-sample-set-from-directory (data-set-path 
                                         label-outputs 
                                         file-transformation)
  "Creates and returns a neural network sample set.  A sample set is a
vector of lists, where each list contains a list of input values and a
list of expected output values. See the definition of class
ALPHA-ENVIRONMENT, in the dc-bianet module, for more information about
the structure of a sample set. Specifically, look at the documentation
of the TRAINING-SET class attribute.

DATA-SET-PATH is the path to a data-set directory that contains
subdirectories that represent labels, and which, in turn, contain
files that represent samples for that label. See the comments at the
beginning of this file for more information about the structure of a
data-set directory.

LABEL-OUTPUTS is a hash table that maps each label string to a list of
floating-point values, which are the expected-output values for that
label. For more information about the LABEL-OUTPUTS hash, see the
documentation for the function LABEL-OUTPUTS, in the labels module.

The LABEL-OUTPUTS hash table could easily be computed from the data in
the directory at DATA-SET-PATH, but any caller is already likely to
have the LABEL-OUTPUTS hash table handy already, so this function
requires the parameter to save itself the work of computing the hash
table again.

FILE-TRANSFORMATION is a function that takes a file path and returns a
vector of input values for the neural network. The function reads the
file and transforms it into a list of normalized floating-point values
that the neural network can use as input. The function should
normalize the values to be between 0.0 and 1.0."
  (loop
    with label-files = (data-set-files data-set-path)
    with total-file-count = (loop 
                              for label being the hash-keys in label-files
                                using (hash-value files)
                              summing (length files))
    with sample-set = (make-array total-file-count
                                  :element-type 'list
                                  :initial-element nil)
    and index = 0
    for label being the hash-keys in label-files
      using (hash-value files)
    for outputs = (gethash label label-outputs)
    do (loop
         for file in files
         for inputs = (funcall file-transformation file)
         do (setf (aref sample-set index) (list inputs outputs))
            (incf index))
    finally (return sample-set)))
