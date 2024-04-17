TESTS_FILE="$(HOME)/common-lisp/dc-bianet/test/dc-bianet-tests.lisp"
LISP=/usr/bin/sbcl
# Reporter can be list dot tap or fiveam.
REPORTER=list
test:
	$(LISP) --eval "(ql:quickload :prove)" \
	  --eval "(ql:quickload :swank)" \
	  --eval "(require :prove)" \
	  --eval "(prove:run #P\"$(TESTS_FILE)\" :reporter :$(REPORTER))" \
	  --non-interactive
