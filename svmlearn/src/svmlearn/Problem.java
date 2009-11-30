package svmlearn;

/**
 * Class representing an optimization problem (a data setting);
 * taken from liblinear; "bias" excluded
 * @author miafranc
 *
 */
public class Problem {
	/** The number of training data */
	public int l;
	/** The number of features (including the bias feature if bias &gt;= 0) */
	public int n;
	/** Array containing the target values */
	public int[] y;
	/** Array of sparse feature nodes */
	public FeatureNode[][] x;
	public void loadProblem(String filename) {

	}
}
