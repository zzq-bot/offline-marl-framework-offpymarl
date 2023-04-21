git clone git@github.com:oxwhirl/smac.git
pip install -e smac/
cp -r patches/smac/* smac/smac/env/starcraft2/maps 
cp patches/smac/SMAC_Maps/* $SC2PATH/Maps/SMAC_Maps/