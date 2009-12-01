package svmlearn;

/**
 * Main class containing the SVM's train and test metods
 * @author miafranc
 *
 */
public class SVM {
	/** Trained/loaded model */
	private Model model;
	/** Regularization parameter */
	private double C = 1;
	/** Tolerance */
	private double tol = 10e-3;
	/** Tolerance */
	private double tol2 = 10e-5;
	/** Number of times to iterate over the alpha's without changing */
	private int maxpass = 10;
	
	/*some global variables of the SMO algorithm*/
	private double Ei, Ej;
	private double ai_old, aj_old;
	private double L, H;
	/* ---------------------------------------- */

	public SVM() {
	}
	/**
	 * Training the SVM.
	 * @param train The training set.
	 */
	public void svmTrain(Problem train) {
		KernelParams p = new KernelParams();
		svmTrain(train, p, 0);
	}
	/**
	 * Training the SVM with specified kernel parameters and algorithm. 
	 * @param train The training set. 
	 * @param p The kernel parameters.
	 * @param alg The chosen algorithm.
	 */
	public void svmTrain(Problem train, KernelParams p, int alg) {
		switch (alg) {
		case 0:
			simpleSMO(train, p);
			break;
		case 1:
			SMO(train, p);
			break;
		}
	}
	/**
	 * Probabilistic (random, simple) SMO
	 * @param train The training set.
	 * @param p The kernel parameters.
	 */
	private void simpleSMO(Problem train, KernelParams p) {
		int pass = 0;
		int alpha_change = 0;
		int i, j;
		double eta;
		//Initialize:
		model = new Model();
		model.alpha = new double [train.l];
		model.b = 0;
		model.params = p;
		model.x = train.x;
		model.y = train.y;
		model.l = train.l;
		model.n = train.n;
		//Main iteration:
		while (pass < maxpass) {
			if (alpha_change > 0)
				System.out.print(".");
			else
				System.out.print("*");
			alpha_change = 0;
			for (i=0; i<train.l; i++) {
				Ei = svmTestOne(train.x[i]) - train.y[i];
				if ((train.y[i]*Ei<-tol && model.alpha[i]<C) || (train.y[i]*Ei>tol && model.alpha[i]>0)) {
					j = (int)Math.floor(Math.random()*(train.l-1));
					j = (j<i)?j:(j+1);
					Ej = svmTestOne(train.x[j]) - train.y[j];
					ai_old = model.alpha[i];
					aj_old = model.alpha[j];
					L = computeL(train.y[i], train.y[j]);
					H = computeH(train.y[i], train.y[j]);
					if (L == H) //next i
						continue;
					eta = 2*kernel(train.x[i],train.x[j])-kernel(train.x[i],train.x[i])-kernel(train.x[j],train.x[j]);
					if (eta >= 0) //next i
						continue;
					model.alpha[j] = aj_old - (train.y[j]*(Ei-Ej))/eta;
					if (model.alpha[j] > H)
						model.alpha[j] = H;
					else if (model.alpha[j] < L)
						model.alpha[j] = L;
					if (Math.abs(model.alpha[j]-aj_old) < tol2) //next i
						continue;
					model.alpha[i] = ai_old + train.y[i]*train.y[j]*(aj_old-model.alpha[j]);
					computeBias(model.alpha[i], model.alpha[j], train.y[i], train.y[j], 
							kernel(train.x[i], train.x[i]), kernel(train.x[j], train.x[j]), 
							kernel(train.x[i], train.x[j]));
					alpha_change++;
				}
			}
			if (alpha_change == 0)
				pass++;
			else
				pass = 0;
		}
		System.out.println();
	}
	/**
	 * Computes L.
	 * @param yi
	 * @param yj
	 * @return Returns L.
	 */
	private double computeL(int yi, int yj) {
		double L = 0;
		if (yi != yj) {
			L = Math.max(0, -ai_old+aj_old);
		} else {
			L = Math.max(0, ai_old+aj_old-C);
		}
		return L;
	}
	/**
	 * Computes H.
	 * @param yi
	 * @param yj
	 * @return Returns H.
	 */
	private double computeH(int yi, int yj) {
		double H = 0;
		if (yi != yj) {
			H = Math.min(C, -ai_old+aj_old+C);
		} else {
			H = Math.min(C, ai_old+aj_old);
		}
		return H;
	}
	/**
	 * Computes the bias and stores in model.b.
	 * @param ai
	 * @param aj
	 * @param yi
	 * @param yj
	 * @param kii
	 * @param kjj
	 * @param kij
	 */
	private void computeBias(double ai, double aj, int yi, int yj, double kii, double kjj, double kij) {
		double b1 = model.b - Ei - yi*(ai-ai_old)*kii - yj*(aj-aj_old)*kij;
		double b2 = model.b - Ej - yi*(ai-ai_old)*kij - yj*(aj-aj_old)*kjj;
		if (0 < ai && ai<C)
			model.b = b1;
		else if (0 < aj && aj < C)
			model.b = b2;
		else
			model.b = (b1+b2)/2;		
	}
	/**
	 * The famous SMO algorithm.
	 * @param train The training set.
	 * @param p The kernel parameters.
	 */
	private void SMO(Problem train, KernelParams p) {
		
	}
	/**
	 * Test a whole data set
	 * @param test The test data
	 * @return An array of -1 and 1's
	 */
	public int [] svmTest(Problem test) {
		if (test == null) 
			return null;
		int [] pred = new int[test.l];
		for (int i=0; i<test.l; i++) {
			pred[i] = (svmTestOne(test.x[i])<0?-1:1);
		}
		return pred;
	}
	/**
	 * Test one example
	 * @param x The test example
	 * @return Class of x: -1 or 1
	 */
	public double svmTestOne(FeatureNode [] x) {
		double f = 0;
		for (int i=0; i<model.l; i++) {
			f += model.alpha[i]*model.y[i]*kernel(x, model.x[i]);
		}
		return f+model.b;
	}
	/**
	 * Based on the kernel parameters/settings of the model,
	 * calculates the kernel value between two points.
	 * @param x
	 * @param z
	 * @return Kernel value between x and z.
	 */
	private double kernel(FeatureNode [] x, FeatureNode [] z) {
		double ret = 0;
		switch (model.params.kernel) {
		case 0: //user defined
			break;
		case 1: //linear
			ret = Kernel.kLinear(x, z);
			break;
		case 2: //polynomial
			ret = Kernel.kPoly(x, z, model.params.a, model.params.b, model.params.c);
			break;
		case 3: //gaussian
			ret = Kernel.kGaussian(x, z, model.params.a);
			break;
		case 4: //tanh
			ret = Kernel.kTanh(x, z, model.params.a, model.params.b);
			break;
		}
		return ret;
	}
	public Model getModel() {
		return model;
	}
	public void setModel(Model m) {
		model = m;
	}
	public double getC() {
		return C;
	}
	public void setC(double C) {
		this.C = C;
	}
	public double getTolerance() {
		return tol;
	}
	public void setTolerance(double tol) {
		this.tol = tol; 
	}
	public double getTolerance2() {
		return tol2;
	}
	public void setTolerance2(double tol) {
		this.tol2 = tol;
	}
	public int getMaxPass() {
		return maxpass;
	}
	public void setMaxPass(int p) { 
		maxpass = p;
	}
	public void setParameters(double C, double tol, double tol2, int maxpass) {
		this.C = C;
		this.tol = tol;
		this.tol2 = tol2;
		this.maxpass = maxpass;
	}
}
