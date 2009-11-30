package svmlearn;

/**
 * Class representing an optimization problem (a data setting)
 * Taken from liblinear; "bias" excluded
 * @author miafranc
 *
 */
public class Problem {
	/** the number of training data */
	public int l;
	/** the number of features (including the bias feature if bias &gt;= 0) */
	public int n;
	/** an array containing the target values */
	public int[] y;
	/** array of sparse feature nodes */
	public FeatureNode[][] x;
}
