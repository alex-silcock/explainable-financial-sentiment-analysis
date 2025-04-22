#!/bin/bash

hare reserve /mnt/faster0/as4387
hare build -t as4387/pt-plus-libs .
hare release /mnt/faster0/as4387
exit