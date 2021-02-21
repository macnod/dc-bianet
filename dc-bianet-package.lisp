;;;; dc-bianet-package.lisp

(defpackage :dc-bianet
  (:use :cl :sb-thread :sb-concurrency :dc-dlist :cl-cpus :clog :clog-user)

  (:export 
   t-cx
   t-neuron
   t-net))
