;;;; dc-bianet-package.lisp

(defpackage :dc-bianet
  (:use :cl
        :cl-cpus
        :cl-ppcre
        :dc-dlist
        :sb-concurrency
        :sb-thread
        :zpng
        :vgplot
        :dc-eclectic
        :dc-ds
        :vecto)
  (:local-nicknames (:t :transducers))
  (:shadow :range))
