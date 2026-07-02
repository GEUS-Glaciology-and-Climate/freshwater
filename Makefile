# Freshwater discharge pipeline.
#
# Stages are tracked with stamp files in stamps/. Each stamp depends on the
# scripts that build it AND on the stamps of the stages it needs, so any
# target bootstraps its prerequisites (e.g. `make RCM` on a clean tree builds
# G, sob and err first). Each stage tees its output to logs/<stage>.log.
#
# Required environment:
#   DATADIR   path to the input datasets (MAR, RACMO, BedMachine, ...)
# Override if needed:
#   PYTHON    interpreter for the freshwater conda env (default below)
#   PROJ_LIB  PROJ data dir (default /usr/share/proj, set in scripts/common.sh)
#
# Common entry points:
#   make            # build everything (data product + k_basin_change figures)
#   make RCM        # build one stage and whatever it depends on
#   make update YEAR=2027
#   make clean_RCM  # drop RCM/coverage/output to force a rebuild

# Pipe failures (e.g. a stage failing before `tee`) must fail the recipe.
SHELL       := /bin/bash
.SHELLFLAGS := -o pipefail -c

SCRIPTS := scripts
PYTHON  ?= /home/shl/miniconda3/envs/freshwater/bin/python
GRASSDB := ./G
GRASS   := grass $(GRASSDB)/PERMANENT --exec
LOGDIR  := logs
YEAR    ?= 2026

# Make DATADIR/PYTHON/PROJ_LIB visible to every script and sub-make.
export DATADIR
export PYTHON
export PROJ_LIB

.DEFAULT_GOAL := all

.PHONY: all preflight G sob k_basin_change err RCM coverage output update dist \
        clean_all clean_RCM clean_sob

all: stamps/output stamps/k_basin_change

# Fail fast with a clear message before any heavy work starts.
preflight:
	@if [ -z "$${DATADIR:-}" ]; then \
	  echo "ERROR: DATADIR is unset. Fix with: export DATADIR=/path/to/data"; exit 1; fi
	@if ! [ -x "$(PYTHON)" ] && ! command -v "$(PYTHON)" >/dev/null 2>&1; then \
	  echo "ERROR: PYTHON ($(PYTHON)) not found. Override with: make PYTHON=/path/to/python"; exit 1; fi
	@mkdir -p $(LOGDIR) stamps

# Phony convenience aliases — each delegates to the real stamp target.
G:             stamps/G
sob:           stamps/sob
k_basin_change: stamps/k_basin_change
err:           stamps/err
RCM:           stamps/RCM
coverage:      stamps/coverage
output:        stamps/output

# GRASS location (EPSG:3413)
stamps/G: | preflight
	@if [ ! -d $(GRASSDB) ]; then \
	  grass -e -c EPSG:3413 $(GRASSDB) 2>&1 | tee $(LOGDIR)/G.log; fi
	@touch $@

# basins, outlets, and streams
stamps/sob: $(SCRIPTS)/common.sh $(SCRIPTS)/import.sh $(SCRIPTS)/sob.sh stamps/G | preflight
	( $(GRASS) $(SCRIPTS)/import.sh && \
	  $(GRASS) $(SCRIPTS)/sob.sh ) 2>&1 | tee $(LOGDIR)/sob.log
	@touch $@

# how do basins change under different SOB routing? (figures only)
stamps/k_basin_change: $(SCRIPTS)/common.sh $(SCRIPTS)/k_basin_change.sh $(SCRIPTS)/k_basin_change.py stamps/sob | preflight
	$(GRASS) $(SCRIPTS)/k_basin_change.sh 2>&1 | tee $(LOGDIR)/k_basin_change.log
	@touch $@

# 2D EPSG area error correction rasters
stamps/err: $(SCRIPTS)/common.sh $(SCRIPTS)/area_error.sh stamps/G | preflight
	$(GRASS) $(SCRIPTS)/area_error.sh 2>&1 | tee $(LOGDIR)/err.log
	@touch $@

# RCM runoff partitioned across basins (needs basins from sob + area error)
stamps/RCM: $(SCRIPTS)/common.sh \
            $(SCRIPTS)/mar_preprocess.py $(SCRIPTS)/rcm_mar.sh \
            $(SCRIPTS)/racmo_preprocess.py $(SCRIPTS)/rcm_racmo.sh \
            stamps/sob stamps/err | preflight
	@mkdir -p dat
	( $(PYTHON) $(SCRIPTS)/mar_preprocess.py && \
	  $(GRASS) $(SCRIPTS)/rcm_mar.sh && \
	  $(PYTHON) $(SCRIPTS)/racmo_preprocess.py && \
	  $(GRASS) $(SCRIPTS)/rcm_racmo.sh ) 2>&1 | tee $(LOGDIR)/RCM.log
	@touch $@

# per-basin RCM coverage fraction
stamps/coverage: $(SCRIPTS)/common.sh $(SCRIPTS)/coverage_calc.sh $(SCRIPTS)/coverage_csv2nc.py stamps/RCM | preflight
	( $(GRASS) $(SCRIPTS)/coverage_calc.sh && \
	  $(PYTHON) $(SCRIPTS)/coverage_csv2nc.py ) 2>&1 | tee $(LOGDIR)/coverage.log
	@touch $@

# end-user NetCDF data product (scales discharge by coverage)
stamps/output: $(SCRIPTS)/bsv2netcdf.py stamps/coverage | preflight
	parallel "$(PYTHON) $(SCRIPTS)/bsv2netcdf.py {}" ::: $$(ls dat/*_runoff_*.bsv) \
	  2>&1 | tee $(LOGDIR)/output.log
	@touch $@

# Fetch and reprocess a single MAR year (e.g. make update YEAR=2027)
update: | preflight
	wget http://ftp.climato.be/fettweis/tmp/ken/1km/MARv3.14-daily-ERA5-$(YEAR).nc \
	  -O $(DATADIR)/MAR/3.14-daily-1km/MARv3.14-daily-ERA5-$(YEAR).nc
	rm -f tmp/MAR_runoff_*_$(YEAR).nc
	$(PYTHON) $(SCRIPTS)/mar_preprocess.py
	rm -f dat/MAR_runoff_*_$(YEAR)_100.bsv
	$(GRASS) $(SCRIPTS)/rcm_mar.sh
	parallel "$(PYTHON) $(SCRIPTS)/bsv2netcdf.py {}" ::: $$(ls dat/*_runoff_*$(YEAR)*.bsv)

dist: freshwater.zip
freshwater.zip: all
	zip -r freshwater.zip freshwater

clean_all:
	rm -fR G tmp dat freshwater freshwater.zip stamps $(LOGDIR)

clean_RCM:
	rm -fR G/MAR G/RACMO stamps/RCM stamps/coverage stamps/output

clean_sob:
	rm -fR G/land_* G/ice_* stamps/sob stamps/k_basin_change stamps/err stamps/RCM stamps/coverage stamps/output
