package svmlearn;

public class EvalMeasures {
	double [] tp;
	double [] fp;
	double [] fn;
	Problem p;
	int [] predicted;
	int catnum;
	int computed;
	
	public EvalMeasures(Problem p, int [] predicted, int catnum) {
		if (predicted.length != p.l) {
			System.err.println("Length error!");
			return;
		}
		this.p = p;
		this.predicted = predicted;
		this.catnum = catnum;
		computed = 0;
	}
	public void countStuff() {
		tp = new double [catnum];
		fp = new double [catnum];
		fn = new double [catnum];
		for (int i=0; i<p.l; i++) {
			if (p.y[i] == predicted[i]) 
				tp[p.y[i]]++;
			else {
				fp[predicted[i]]++;
				fn[p.y[i]]++;
			}
		}
		computed = 1;
	}
	public double Accuracy() {
		int ret = 0;
		for (int i=0; i<p.l; i++) {
			if (p.y[i] == predicted[i]) {
				ret++;
			}
		}
		return (double)ret/p.l;
	}
	public double MacroF(double beta) {
		if (computed == 0) countStuff();
		double mprec = MacroPrecision(); 
		double mrec = MacroRecall();
		return ((beta*beta + 1)*mprec*mrec/(beta*beta*mprec + mrec));
	}
	public double MacroF() {
		return MacroF(1);
	}
	public double MacroPrecision() {
		if (computed == 0) countStuff();
		double mprec = 0;
		for (int i=0; i<catnum; i++) {
			if ((tp[i] != 0) && (tp[i]+fp[i] != 0))
				mprec += tp[i]/(tp[i]+fp[i]);
		}
		return mprec/catnum;
	}
	public double MacroRecall() {
		if (computed == 0) countStuff();
		double mrec = 0;
		for (int i=0; i<catnum; i++) {
			if ((tp[i] != 0) && (tp[i]+fn[i] != 0))
				mrec += tp[i]/(tp[i]+fn[i]);
		}
		return mrec/catnum;
	}
}
