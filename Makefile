SCRIPTS = scripts
PYTHON  = /home/shl/miniconda3/envs/freshwater/bin/python

.PHONY: G all sob k_basin_change err RCM coverage output update dist clean_all clean_RCM clean_sob FORCE

freshwater.zip: G all

G:
	grass -e -c EPSG:3413 ./G

all:
	$(MAKE) sob
	$(MAKE) k_basin_change
	$(MAKE) err
	$(MAKE) RCM
	$(MAKE) coverage
	$(MAKE) output

# Phony convenience aliases — each delegates to the real stamp target.
sob:           stamps/sob
k_basin_change: stamps/k_basin_change
err:           stamps/err
RCM:           stamps/RCM
coverage:      stamps/coverage
output:        stamps/output

stamps/sob: $(SCRIPTS)/import.sh $(SCRIPTS)/sob.sh
	# basins, outlets, and streams
	grass ./G/PERMANENT --exec $(SCRIPTS)/import.sh
	grass ./G/PERMANENT --exec $(SCRIPTS)/sob.sh
	@mkdir -p stamps && touch $@

stamps/k_basin_change: $(SCRIPTS)/k_basin_change.sh $(SCRIPTS)/k_basin_change.py
	# how do basins change under different SOB routing?
	grass ./G/PERMANENT --exec $(SCRIPTS)/k_basin_change.sh
	@mkdir -p stamps && touch $@

stamps/err: $(SCRIPTS)/area_error.sh
	# 2D EPSG area error
	grass ./G/PERMANENT --exec $(SCRIPTS)/area_error.sh
	@mkdir -p stamps && touch $@

stamps/RCM: $(SCRIPTS)/mar_preprocess.py $(SCRIPTS)/rcm_mar.sh \
            $(SCRIPTS)/racmo_preprocess.py $(SCRIPTS)/rcm_racmo.sh
	# RCM
	mkdir -p dat
	$(PYTHON) $(SCRIPTS)/mar_preprocess.py
	grass ./G/PERMANENT --exec $(SCRIPTS)/rcm_mar.sh
	$(PYTHON) $(SCRIPTS)/racmo_preprocess.py
	grass ./G/PERMANENT --exec $(SCRIPTS)/rcm_racmo.sh
	@mkdir -p stamps && touch $@

stamps/coverage: $(SCRIPTS)/coverage_calc.sh $(SCRIPTS)/coverage_csv2nc.py
	# coverage
	grass ./G/PERMANENT --exec $(SCRIPTS)/coverage_calc.sh
	$(PYTHON) $(SCRIPTS)/coverage_csv2nc.py
	@mkdir -p stamps && touch $@

stamps/output: $(SCRIPTS)/bsv2netcdf.py
	# create end-user data product
	parallel "$(PYTHON) $(SCRIPTS)/bsv2netcdf.py {}" ::: $$(ls dat/*_runoff_*.bsv)
	@mkdir -p stamps && touch $@

update:
	wget http://ftp.climato.be/fettweis/tmp/ken/1km/MARv3.14-daily-ERA5-2026.nc -O ${DATADIR}/MAR/3.14-daily-1km/MARv3.14-daily-ERA5-2026.nc
	rm -f tmp/MAR_runoff_*_2026.nc
	$(PYTHON) $(SCRIPTS)/mar_preprocess.py
	rm dat/MAR_runoff_*_2026_100.bsv
	grass ./G/PERMANENT --exec $(SCRIPTS)/rcm_mar.sh
	parallel "$(PYTHON) $(SCRIPTS)/bsv2netcdf.py {}" ::: $$(ls dat/*_runoff_*2026*.bsv)

dist: freshwater.zip
	zip -r freshwater.zip freshwater

FORCE:

clean_all:
	rm -fR G tmp dat freshwater freshwater.zip stamps

clean_RCM:
	rm -fR G/MAR G/RACMO stamps/RCM stamps/coverage stamps/output

clean_sob:
	rm -fR G/land_* G/ice_* stamps/sob stamps/k_basin_change stamps/err stamps/RCM stamps/coverage stamps/output
