#!/bin/sh

cd CMG                                                        && \
curl -sS http://www.cs.cmu.edu/~jkoutis/CMG/CMG.zip > CMG.zip && \
unzip CMG.zip                                                 && \
rm CMG.zip                                                    && \
matlab -nodesktop -r 'MakeCMG; exit;'                         && \
cd ../
