.PHONY: install data daily analysis visualize refs supplementary strategy flowchart

install:
	python3 -m pip install -r requirements.txt

data:
	python3 scripts/01_fetch_monthly_data.py

daily:
	python3 scripts/02_build_event_window.py

analysis:
	python3 scripts/03_run_analysis_tables.py

visualize:
	python3 scripts/04_plot_core_figures.py

refs:
	python3 scripts/05_plot_reference_figures.py

supplementary:
	python3 scripts/06_plot_supplementary_figures.py

strategy:
	python3 scripts/07_build_strategy_tables.py

flowchart:
	python3 scripts/08_render_conclusion_flowchart.py
