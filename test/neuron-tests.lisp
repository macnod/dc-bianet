(require :dc-eclectic)
(require :dc-bianet)
(require :prove)

(defpackage :neuron-tests
  (:use :cl :dc-eclectic :dc-bianet :prove))

(in-package :neuron-tests)

(plan 1)

(let ((neuron
