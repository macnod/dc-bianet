;;;; dc-bianet-package.lisp

(defpackage :dc-bianet
  (:use :cl :sb-thread :dc-dlist)
  (:export t-cx t-neuron t-net feed train))


