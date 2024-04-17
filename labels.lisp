;; This component contains code to deal with labels, except with code
;; to actually retrieve labels from files or directories, which can be
;; found in the files component.

(in-package :dc-bianet)

(defclass output-labels ()
  ((data-set-path
    :accessor data-set-path
    :initarg :data-set-path
    :initform (error "data-set-path is required")
    :type (or pathname string)
    :documentation
    "Required. The path to the data-set directory.See the files module for
a description of the data-set directory structure.")
   (label-count 
    :accessor label-count
    :initarg :label-count
    :initform (make-hash-table)
    :type hash-table
    :documentation 
    "A hash table mapping each label name to the number of samples for that
label.")
   (label-vector 
    :accessor label-vector
    :initarg :label-vector
    :initform (vector)
    :type vector
    :documentation 
    "A vector of label names, with the labels in alphabetical order. This
can be useful when you need to associate a label with a neural network
output index.")
   (label-index 
    :accessor label-index
    :initarg :label-index
    :initform (make-hash-table)
    :type hash-table
    :documentation 
    "A hash table mapping each label name to the index of that label in the
label-vector.  This index exactly corresponds to the index of the
neural newtork output index.")
   (label-outputs 
    :accessor label-outputs
    :initarg :label-outputs
    :initform (make-hash-table)
    :type hash-table
    :documentation 
    "A hash table mapping each label name to a list of desired neural
network output values. All of the values in the list are 0.0, except
for the value at the index corresponding to the label, which is 1.0."))
  (:documentation 
   "Holds information about labels so that training sets can be created
and so that inference results, which consist of a list of
floating-point output values, can be converted into a label. This
class also allows queries about the number of samples for each label."))

(defmethod initialize-instance :after ((output-labels output-labels) &key)
  (when (zerop (hash-table-count (label-count output-labels)))
    (let* ((path (data-set-path output-labels))
           (label-count (make-label-count path))
           (label-vector (make-label-vector label-count))
           (label-index (make-label-index label-count))
           (label-outputs (make-label-outputs label-count)))
      (setf (label-count output-labels) label-count
            (label-vector output-labels) label-vector
            (label-index output-labels) label-index
            (label-outputs output-labels) label-outputs))))

(defun make-label-count (data-set-path)
  "Returns a hash table where each entry consists of a label and the
number of samples available for that label. 

The function first determines if the DATA-SET-PATH points to a file or
a directory.

If DATA-SET-PATH points to a file, the function assumes that the file
is a CSV file and scans the file, retrieving the first element of each
line, which should be the label. The function stores each label in the
hash table result, incrementing the label's associated value in the
hash to represent the number of times the function has seen the label
in the CSV file.

The function assumes that the directory is a data-set directory (see
the files module for a description of a data-set directory) and
considers all subdirectory names to be labels. The function then
counts the files in each subdirectory to determine the count of
samples for each label, and builds the resulting hash table
accordingly."
  (case (path-type data-set-path)
    (:file 
     (csv-label-sample-size data-set-path))
    (:directory 
     (label-sample-size data-set-path))
    (:not-found 
     (error "File not found: ~a" data-set-path))))

(defgeneric make-label-vector (object)
  (:documentation "Returns a vector with labels, sorted in
alphabetical order.

If OBJECT is a list of labels, this function sorts the list and
returns a vector created from the sorted list.

If OBJECT is a directory, this function assumes the directory to have
the structure of a data-set directory. A data-set directory contains
subdirectories that represent the labels.  See the files module for
more information on what a data-set directory looks like. This
function returns a vector of labels (names of the subdirectories of
OBJECT), sorted alphabetically.

If OBJECT is a file, this function assumes the file to be formatted as
a CSV, builds a list of the first element of each row, and creates a
vector from the sorted distinct elements of the list.  For performance
reasons, the labels can't have commas.

If OBJECT is a hash table, then the resulting vector is built by taking 
the hash table's keys and sorting them.")
  (:method ((list list))
    (map 'vector 'identity (sort list #'string<)))
  (:method ((table hash-table))
    (make-label-vector (hash-keys table)))
  (:method ((path string))
    (make-label-vector (get-labels path :sorted nil)))
  (:method ((path pathname))
    (make-label-vector (format nil "~a" path))))

(defgeneric make-label-index (labels-data)
  (:documentation "Returns a hash table with entries label -> {neural
network output index}.  This is the inverse of the vector that
MAKE-LABEL-VECTOR produces, where you can retrieve a label by an
index. MAKE-LABEL-INDEX produces a hash that allows you to retrieve an
index given the label.

The neural network output index is computed by sorting the labels
alphabetically and then assigning an index to each label, starting
from 0 and incrementing by 1 for each label.

The parameter can be a list, a hash table, or a data-set directory.
See the comments at the beginning of the files module for a
description of data-set directories.

If LABELS-DATA is a directory (a string or a pathname), the function
assumes that the directory is a data-set directory. See the comments
at the top of the files component for a description of a data-set
directory. They data-set directory's subdirectory names become the
labels.

If LABELS-DATA is a file, the function assumes that the file is a CSV
file.

If LABELS-DATA is hash table, the keys of the hash table become the
labels.")
  (:method ((labels-list list))
    (hashify-list (sort labels-list #'string<) :method :index))
  (:method ((label-count hash-table))
    (make-label-index (hash-keys label-count)))
  (:method ((data-set string))
    (make-label-index (get-labels data-set :sorted nil)))
  (:method ((data-set pathname))
    (make-label-index (format nil "~a" data-set))))

(defgeneric make-label-outputs (labels-data)
  (:documentation "Returns a hash table with entries that look like
this:

  label-1 -> (1.0 0.0 0.0)
  label-2 -> (0.0 1.0 0.0)
  label-3 -> (0.0 0.0 1.0)

The keys are strings and the values are lists of floating-point
numbers.  This table is useful for creating the example vectors needed
to train a neural network. The outputs correspond to the desired
activation values for the neural network's outputs. This is useful for
training the neural network. However, for inference, we use a vector
of labels. Then, we determine of the index of the neural network's
output with the largest value, and we use that index to find the label
in the vector.

LABELS-DATA can be a list of labels, a vector, or a hash table where
the keys are the labels.

This function assigns the output indexes to the labels in ascending
order, after the labels have been sorted alphabetically.")
  (:method ((labels-list list))
    (loop
      with l = (length labels-list)
      with label-outputs = (make-hash-table :test 'equal :size l)
      for label in (sort labels-list #'string<)
      for index = 0 then (1+ index)
      for outputs = (loop for i from 0 below l
                         collect (if (= i index) 1.0 0.0))
      do (setf (gethash label label-outputs) outputs)
      finally (return label-outputs)))
  (:method ((table hash-table))
    (make-label-outputs (hash-keys table)))
  (:method ((data-set string))
    (make-label-outputs (get-labels data-set :sorted nil)))
  (:method ((data-set pathname))
    (make-label-outputs (format nil "~a" data-set)))
  (:method ((labels-vector vector))
    (make-label-outputs (map 'list 'identity labels-vector))))

(defun outputs-label (labels-vector outputs-list)
  "Returns the label, a string from LABELS-VECTOR, that corresponds to
OUTPUT-LIST, a list of floating-point values. LABELS-VECTOR and
OUTPUT-LIST must have the same length. The function first computes the
index of the greatest OUTPUT-LIST value, then uses that index to
lookup the corresponding label in LABELS-VECTOR."
  (aref labels-vector (index-of-max outputs-list)))
