(ds (list :map
          :db "bianet"
          :username "bianet"
          :password "weasel"
          :host "localhost"
          :retry-count 3
          :retry-sleep 1
          :retry-sleep-factor 3
          :log-function (lambda (&rest messages)
                          (with-open-file (out (join-paths *log-folder* "db.log")
                                               :direction :output
                                               :if-exists :append
                                               :if-does-not-exist :create)
                            (write-line (format nil "~{~a~}" messages) out)))))
                                             
