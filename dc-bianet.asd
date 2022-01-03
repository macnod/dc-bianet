;;;; dc-bianet.asd

(ql:quickload '(:cl-cpus :zpng :png-read :vgplot :cl-ppcre))
(asdf:defsystem #:dc-bianet
  :description "Flexible multilayer backprop neural network simulation."
  :author "Donnie Cameron <macnod@gmail.com>"
  :license "MIT License"
  :depends-on (#:sb-concurrency 
               #:dc-dlist 
               #:cl-cpus 
               #:zpng 
               #:png-read
               #:vgplot
               #:cl-ppcre
               #:dc-utilities
               #:swank
               #:dc-db
							 #:mcclim
							 #:clim-lisp)
  :serial t
  :components ((:file "dc-bianet-package")
               (:file "dc-bianet")))
