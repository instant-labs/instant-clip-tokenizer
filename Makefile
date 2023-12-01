ifeq ($(shell uname), Darwin)
	PY_EXT := dylib
else
	PY_EXT := so
endif

test-python:
	cargo build --release
	cp target/release/libinstant_clip_tokenizer.$(PY_EXT) instant-clip-tokenizer-py/test/instant_clip_tokenizer.so
	PYTHONPATH=instant-clip-tokenizer-py/test/ python3 -m test

validate:
	cargo build --release
	cp target/release/libinstant_clip_tokenizer.$(PY_EXT) scripts/instant_clip_tokenizer.so
	PYTHONPATH=scripts/ python3 -m validate scripts/Train_GCC-training.tsv
