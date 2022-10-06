#!/bin/bash

case $1 in
	clean)
		find . -name "*.fls" -type f -delete
		find . -name "*.fdb_latexmk" -type f -delete
		find . -name "*.log" -type f -delete
		find . -name "*.gz" -type f -delete
		find . -name "*.aux" -type f -delete
	;;
esac
