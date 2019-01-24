#!/bin/sh

cd minFunc/                                                                           && \
curl -sS http://www.cs.ubc.ca/~schmidtm/Software/minFunc_2012.zip > minFunc_2012.zip  && \
unzip minFunc_2012.zip                                                                && \
rm minFunc_2012.zip                                                                   && \
cd minFunc_2012/                                                                      && \
matlab -nodesktop -r 'mexAll; exit;'                                                  && \
cd ../../
