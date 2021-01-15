;; For a network with topology 2 9 5 2 with each hidden layer count
;; including a bias
(defparameter *random-weights* 
  '(0.76202846 0.2605176 -0.2796769 -0.6233269 -0.33678865 -0.6740299
    -0.64292073 0.7174244 -0.028132915 -0.8992507 -0.0011966825
    0.34103918 -0.4258262 -0.8837741 -0.48971558 -0.7873071 -0.012343109
    0.64152193 0.52223957 -0.025228858 -0.56446487 -0.35145628 0.8890362
    -0.59927285 0.5527961 -0.023116112 0.18404484 -0.23808748 0.13314569
    -0.72438073 0.5054325 -0.79919636 -0.31140143 -0.33236152 0.40674758
    0.2879107 -0.8736289 -0.29396838 0.557642 -0.30434787 0.16958594
    -0.7348785 -0.37952358 -0.07510853 0.61446214 -0.6234709 -0.12174332
    0.56028616 -0.4402668 -0.31196964 0.6199081 0.42829323 -0.13455272
    -0.5320198 -0.7641469 0.76636744 0.71680963 0.8683692 0.36208808
    -0.6481888 -0.5203867 -0.06542981))

(defparameter *good-weights*
  '(-0.6433538 0.68500024 0.9822374 -0.4984808 -0.5689665 -0.8605004
    1.1787753 1.1559794 -0.15542477 -0.31207344 -1.2366722
    -0.025307178 -0.7189528 0.4733417 2.4722044 -1.1714686 -0.52969205
    -0.6567384 0.13536811 0.57761407 -0.29241186 -0.04495895
    -0.86167663 0.79770106 0.16346407 0.20183969 -1.4162768 0.32847872
    -0.5300661 -0.8352787 -0.14602566 -0.55944765 0.113309145
    0.5727426 0.2520963 -0.15252995 -0.5773908 0.46699855 -0.70066565
    -0.3229912 -0.7721258 -0.011586519 -2.456867 -0.47767976
    0.24322379 -0.83743286 -1.4930372 0.7393421 -0.8093518 -0.17775796
    5.5599403 -0.7999189 -0.60803187 -0.7440909 0.17848374 0.8678242
    -4.354587 4.417412 -0.7760913 0.8459438 2.9272912 -2.9631748))

(defparameter *ann* nil)
(defparameter *bianet* nil)

(defun show-weights ()
  (list :ann (dc-ann::collect-weights *ann*)
        :bianet (dc-bianet::collect-weights *bianet*)
        :diff (loop for ann-weight in (dc-ann::collect-weights *ann*)
                 for bianet-weight in (dc-bianet::collect-weights *bianet*)
                 collect (- bianet-weight ann-weight))))

(defun reset-weights (weights)
      (dc-ann::apply-weights *ann* weights)
      (dc-bianet::apply-weights *bianet* weights)
      (show-weights))

(defun make-nets ()
  (setq *ann* (dc-ann::train-1)
        *bianet* (dc-bianet::create-net 
                  '((:neurons 2 :transfer-key :relu)
                    (:neurons 8 :transfer-key :relu :add-bias t)
                    (:neurons 4 :transfer-key :relu :add-bias t)
                    (:neurons 2 :transfer-key :logistic))
                  :id :net-1))
  (dc-bianet::connect-fully *bianet*)
  (reset-weights *random-weights*))

(make-nets)

(defparameter *set-10k* (dc-ann::circle-data-1hs *ann* 10000))
(defparameter *set-1k* (dc-ann::circle-data-1hs *ann* 1000))
(defparameter *set-10* (dc-ann::circle-data-1hs *ann* 10))

(defun evaluate-inference (set)
  (list :ann (dc-ann::evaluate-training-1hs *ann* set)
        :bianet (dc-bianet::evaluate-inference-1hs *bianet* set)))

(defun feed (inputs)
  (let ((ann-outputs (dc-ann::feed *ann* inputs))
        (bianet-outputs (dc-bianet::infer-frame *bianet* inputs)))
    (list :ann ann-outputs
          :bianet bianet-outputs
          :diff (loop for ann-output in ann-outputs
                   for bianet-output in bianet-outputs
                   collect (- bianet-output ann-output)))))

(defun feed-from-set (set index)
  (feed (car (elt set index))))

(defun show-inputs ()
  (let ((ann-inputs (dc-ann::collect-inputs *ann*))
        (bianet-inputs (dc-bianet::collect-inputs *bianet*)))
    (list :ann ann-inputs 
          :bianet bianet-inputs
          :diff (loop for ann-input in ann-inputs
                   for bianet-input in bianet-inputs
                   collect (- bianet-input ann-input)))))

(defun show-outputs ()
  (let ((ann-outputs (dc-ann::collect-outputs *ann*))
        (bianet-outputs (dc-bianet::collect-outputs *bianet*)))
    (list :ann ann-outputs
          :bianet bianet-outputs
          :diff (loop for ann-output in ann-outputs
                   for bianet-output in bianet-outputs
                     collect (- bianet-output ann-output)))))

(defun show-expected-outputs ()
  (let ((ann-outputs (dc-ann::collect-expected-outputs *ann*))
        (bianet-outputs (dc-bianet::collect-expected-outputs *bianet*)))
    (list :ann ann-outputs
          :bianet bianet-outputs
          :diff (loop for ann-output in ann-outputs
                   for bianet-output in bianet-outputs
                     collect (- bianet-output ann-output)))))

(defun learn-frame (frame)
  (let ((inputs (car frame))
        (outputs (second frame)))
    (dc-ann::learn-vector *ann* inputs outputs)
    (dc-bianet::train-frame *bianet* inputs outputs)
    (show-weights)))

(defun set-expected-outputs (outputs)
  (dc-ann::set-expected-outputs *ann* outputs)
  (dc-bianet::apply-expected-outputs *bianet* outputs)
  (show-expected-outputs))
