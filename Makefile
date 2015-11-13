DIRS = query_dev_props vectorAdd vectorAdd_pinned

compile:
	for i in $(DIRS); do make -C $$i; done

clean:
	rm -rf obj/ bin/
#	for i in $(DIRS); do make -C $$i clean; done
