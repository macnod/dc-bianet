;; This package is for evaluating the performance of functions in dc-bianet. Every
;; function name in this package should start with 'perf-'

(in-package :dc-bianet)

(defun perf-compute-new-weight ()
  (loop
    with test-cases = (loop for a from 1 to 100000
                            collect
                            (let* ((weight (random 1.0))
                                   (delta (/ weight 10.0))
                                   (target-error (/ weight 3))
                                   (source-output (random 1.0))
                                   (learning-rate 0.02)
                                   (momentum 0.1))
                              (list weight delta target-error
                                    source-output learning-rate
                                    momentum)))
    with weight-sum = 0
    and delta-sum = 0
    and calls = 0
    with start-time = (get-internal-real-time)
    for run from 1 to 1000
    do (loop for (weight delta target-error source-output learning-rate momentum)
               in test-cases
             do (multiple-value-bind (new-weight new-delta)
                    (compute-new-weight
                     weight
                     delta
                     target-error
                     source-output
                     learning-rate
                     momentum)
                  (setq weight-sum (+ weight-sum new-weight)
                        delta-sum (+ delta-sum new-delta)
                        calls (1+ calls))))
    finally (return (list :elapsed (format nil "~,4fs" 
                                           (/ (- (get-internal-real-time) 
                                                 start-time)
                                              internal-time-units-per-second))
                          :calls (format nil "~:d" calls)
                          :weight weight-sum
                          :delta delta-sum))))
         
    
    
