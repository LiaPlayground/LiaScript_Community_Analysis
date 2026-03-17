MAIN = main

.PHONY: pdf clean

# LaTeX-Kompilierung (biblatex/biber)
pdf:
	pdflatex -interaction=nonstopmode $(MAIN).tex
	biber $(MAIN)
	pdflatex -interaction=nonstopmode $(MAIN).tex
	pdflatex -interaction=nonstopmode $(MAIN).tex

clean:
	rm -f *.aux *.log *.bbl *.blg *.out *.bcf *.xml *.toc
