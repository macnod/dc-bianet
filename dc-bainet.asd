;;;; dc-bianet.asd

(asdf:defsystem #:dc-ann
  :description "Simple implementation of a multilayer backprop neural network."
  :author "Donnie Cameron <macnod@gmail.com>"
  :license "MIT License"
  :depends-on (#:dc-utilities)
  :serial t
  :components ((:file "dc-bianet-package")
               (:file "dc-bianet")))

