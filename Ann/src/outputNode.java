import java.util.ArrayList;

public class outputNode {
	double U1;
	double U2;
	public outputNode(double U1, double U2) {
		this.U1 = U1;
		this.U2 = U2;
				
	}
	
	
	//part of step 1 is to assign random weights to each input (as you can see in the constructor there are 5 weights to assign)
	public static double[] getWeights (){
		double [] randWeights = new double[2];
		for (int i = 0; i < randWeights.length;i++){
			randWeights[i]= getRandomDouble(-1,1);
		}
		return randWeights;
		
	}
	
	public static double getBias(){
		double Bias = getRandomDouble(-1,1);
		return Bias;
	}
	
	private static double getRandomDouble(double min, double max){
	    double x = (Math.random()*((max-min)+1))+min;
	    return x;
	}
	
	
	public double[] initialiseOutputNodeWeights (int hiddenNodes){
		double[] outputNodeWeights = new double[hiddenNodes];
		
		for (int i = 0; i <hiddenNodes;i++){
			outputNodeWeights[i] = getRandomDouble(-1,1);
		}
		
		return outputNodeWeights;
	}
	
	public double initialiseOutputNodeBias (){
		double outputNodeBias = getRandomDouble(-1,1);
		return outputNodeBias;
	}
	
	
	public static double forwardPass (double[] dataPoint, int hiddenNodes, double[] HNUvalues, double [] outputNodeWeights, double outputNodeBias) {
		
		//now we have calculated the the weighted sums and then the u values for the output node
		double sumValue = 0;
		for (int i =0;i<hiddenNodes;i++){
			sumValue = sumValue + HNUvalues[i]*outputNodeWeights[i];
		}
		double outputNodeWeightedSum = sumValue + outputNodeBias;
		double ONUvalue = sigmoidFunction(outputNodeWeightedSum);			
		return ONUvalue;	
		//System.out.println(outputNodeBias);	
		
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
	
	
	
	public static double backwardPass(double cValue, double ONUvalue){
		//for the output node the delta value is the {(correct value)-(ONUvalue)}*f'(s)
		double ODeltaValue = (cValue-ONUvalue)*diffSigmoidFunction(ONUvalue);
		return ODeltaValue;
	}
	
	
	

}
