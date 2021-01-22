;;;; dc-bianet.asd

(asdf:defsystem #:dc-bianet
  :description "Flexible multilayer backprop neural network simulation."
  :author "Donnie Cameron <macnod@gmail.com>"
  :license "MIT License"
  :depends-on (#:dc-dlist #:cl-ppcre)
  :serial t
  :components ((:file "dc-bianet-package")
               (:file "dc-bianet")))
