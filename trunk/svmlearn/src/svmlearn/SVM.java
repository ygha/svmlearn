package svmlearn;

/**
 * Main class containing the SVM's train and test metods
 * @author miafranc
 *
 */
public class SVM {
	/** Trained/loaded model */
	public Model model;
	public double C = 1;
	public double tol = 0.001;
	public int maxpass = 100;

	public SVM() {
	}
	public void svmTrain(Problem train) {
		KernelParams p = new KernelParams();
		svmTrain(train, p, 0);
	}
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
	public void simpleSMO(Problem train, KernelParams p) {
		int pass = 0;
		int alpha_change = 0;
		int i, j;
		double b1, b2;
		double Ei, Ej;
		double ai_old, aj_old;
		double L, H;
		double eta;
		//Initialize:
		Model model = new Model();
		model.alpha = new double [train.l];
		model.b = 0;
		model.params = p;
		model.x = train.x;
		//Main iteration:
		while (pass < maxpass) {
			alpha_change = 0;
			for (i=0; i<train.l; i++) {
				Ei = svmTestOne(train.x[i]) - train.y[i];
				if ((train.y[i]*Ei<-tol && model.alpha[i]<C) || 
					(train.y[i]*Ei>tol && model.alpha[i]>0)) {
					j = (int)Math.floor(Math.random()*(train.l-1));
					j = (j<i)?j:(j+1);
					Ej = svmTestOne(train.x[j]) - train.y[j];
					ai_old = model.alpha[i];
					aj_old = model.alpha[j];
					if (train.y[i] != train.y[j]) {
						L = Math.max(0, -ai_old+aj_old);
						H = Math.min(C, -ai_old+aj_old+C);
					} else {
						L = Math.max(0, ai_old+aj_old-C);
						H = Math.min(C, ai_old+aj_old);
					}
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
					if (Math.abs(model.alpha[j]-aj_old) < 10e-5) //next i
						continue;
					model.alpha[i] = ai_old + train.y[i]*train.y[j]*(aj_old-model.alpha[j]);
					b1 = model.b - Ei - train.y[i]*(model.alpha[i]-ai_old)*kernel(train.x[i], train.x[i]) -
							train.y[j]*(model.alpha[j]-aj_old)*kernel(train.x[i], train.x[j]);
					b2 = model.b - Ej - train.y[i]*(model.alpha[i]-ai_old)*kernel(train.x[i], train.x[j]) -
							train.y[j]*(model.alpha[j]-aj_old)*kernel(train.x[j], train.x[j]);
					if (0 < model.alpha[i] && model.alpha[i]<C)
						model.b = b1;
					else if (0 < model.alpha[j] && model.alpha[j] < C)
						model.b = b2;
					else
						model.b = (b1+b2)/2;
					alpha_change++;
				}
			}
			if (alpha_change == 0)
				pass++;
			else
				pass = 0;
		}
	}
	public void SMO(Problem train, KernelParams p) {
		
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
	public double kernel(FeatureNode [] x, FeatureNode [] z) {
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
}
