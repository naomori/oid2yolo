#!/bin/bash

for yaml in *.yaml; do
    awk '{ sub("nc: 80", "nc: 2"); print }' ${yaml} > ${yaml}.out
    mv -f ${yaml}.out ${yaml}
done
