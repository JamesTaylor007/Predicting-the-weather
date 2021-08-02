import java.util.ArrayList;

public class hiddenNode {
	double t;
	double w;
	double sr;
	double dsp;
	double drh;
	//constructor which is the input for each hidden node.
	hiddenNode(double Temp, double Wind,double solarRadiation,double airPressure, double Humidity){
		t = Temp;
		w = Wind;
		sr = solarRadiation;
		dsp = airPressure;
		drh = Humidity;
				 			
	}
	


	
	//part of step 1 is to assign random weights to each input (as you can see in the constructor there are 5 weights to assign)
	public static double[] getWeights (){
		double [] randWeights = new double[5];
		for (int i = 0; i < randWeights.length;i++){
			randWeights[i]= getRandomDouble(-2/5,2/5);
		}
		return randWeights;
		
	}
	
	public static double getBias(){
		double Bias = getRandomDouble(-2/5,2/5);
		return Bias;
	}
	
	private static double getRandomDouble(double min, double max){
	    double x = (Math.random()*((max-min)+1))+min;
	    return x;
	}
	
	
	public ArrayList<double[]> initialiseHiddenNodeWeights (int hiddenNodes){
		ArrayList<double[]> hiddenNodeWeights = new ArrayList<double[]>();
	
		for (int i = 0; i <hiddenNodes;i++){
			double [] randWeights = new double[5];
			for (int j = 0; j < randWeights.length;j++){
				randWeights[j]= getRandomDouble(-2/5,2/5);
			}
			hiddenNodeWeights.add(randWeights);
		}
		return hiddenNodeWeights;
	}
	
	public ArrayList<Double> initialiseHiddenNodeBias (int hiddenNodes){
		ArrayList<Double> hiddenNodeBias = new ArrayList<Double>();
		for (int i = 0; i <hiddenNodes;i++){
			hiddenNodeBias.add(getRandomDouble(-2/5,2/5));		
		}
		return hiddenNodeBias;
		
	}
	
	
	
	public static double[] forwardPass (double[] dataPoint, int hiddenNodes, ArrayList<double[]>hiddenNodeWeights, ArrayList<Double> hiddenNodeBias) {
		//so the point of the forward pass is to calculate the weight sums and the activations for every node
		//starting with the hidden nodes
		
		//initialising our arrays with the correct size
		double [] hiddenNodeWeightedSums = new double[hiddenNodes];
		double [] HNUvalues = new double [hiddenNodes];
		//looping through the hidden nodes
		for (int i = 0; i < hiddenNodes;i++){
			double sumValue =0; //keeping track of the sum for our weighted sum value
			//looping through each input
			for (int j = 0; j<dataPoint.length; j++){
				//calculated the weighted sum for each
				sumValue = sumValue + dataPoint[j]*hiddenNodeWeights.get(i)[j];
			}
			//assigning the value to the correct index in the correct array
			hiddenNodeWeightedSums[i] = sumValue + hiddenNodeBias.get(i);
			HNUvalues[i] = sigmoidFunction(hiddenNodeWeightedSums[i]);
		}
		
		return HNUvalues;		
		
	}
	
	public static double sigmoidFunction(double x){
		/*
		 * This function is the activation function which I have choosen to use the sigmoid function. 
		 * it takes an input which is a double called x
		 * and outputs the results of F(x).
		 */
		
		double Fx = 1/(1+Math.exp(-x));
		return Fx;		
	}
	
	public static double diffSigmoidFunction(double x){
		
		/*
		 * This function is the differential of the activation function which I have choosen to use the sigmoid function. 
		 * it takes an input which is a double called x
		 * and outputs the results of F'(x).
		 */
	
		double diffFx = (1/(1+Math.exp(-x)))*(1 - (1/(1+Math.exp(-x))));
		return diffFx;
	}
	
	
	public static double[] backwardPass(double cValue, int hiddenNodes, double[] outputNodeWeights, double ODeltaValue, double[] HNUvalues){
		double[] HDeltaValues = new double[hiddenNodes];
		//now the delta values for the hidden nodes
		for (int i = 0; i < hiddenNodes; i++){
			HDeltaValues[i] = outputNodeWeights[i]*ODeltaValue*diffSigmoidFunction(HNUvalues[i]);
			
		}
		
		return HDeltaValues;
	}
	
	
}
