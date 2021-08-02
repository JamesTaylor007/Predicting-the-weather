import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.lang.Math;
public class Ann2 {
	//this is some variables we will be using later throughout our functions:
	//the step function (ie p= 0.1)

	


	
	
	public static void main(String[] args) throws IOException {
		int epochs = 100000;
		
		//this is some variables we will be using later throughout our functions:
		//the step function (ie p= 0.1)
		double p = 0.025;
		int hiddenNodes = 10;
	
		
		//the hidden nodes lists ie {[w1,w2,w3,w4,w5],[w1,w2,w3,w4,w5]} where
		//{entry1,entry2} entry x being node x and the weights for each input into the node. 
		//there will be as many entrys as there are hidden nodes, the bias is in the same format
		ArrayList<double[]> hiddenNodeWeights;
		ArrayList<Double> hiddenNodeBias;
		double[] hiddenNodeWeightedSums;
		double[] HNUvalues; //u=f(Sj)
		double [] HDeltaValues;
		
		//now for the output node
		double [] outputNodeWeights;
		double outputNodeBias;
		double outputNodeWeightedSum;
		double ONUvalue = 0; //output node u= f(Sj)
		double ODeltaValue;
		
		ArrayList<Double> validationError = new ArrayList<Double>();
		validationError.add((double) 100);
		// TODO Auto-generated method stub
		String filePath = new File("").getAbsolutePath();
		ArrayList<String> testData = importData(filePath + "\\" + "aiData.txt");
		//System.out.println(testData);
		
		//the first thing is to initiase the weights and then do a forward pass
		
		
		//assigning random values to the hidden weight
		
		hiddenNodeWeights = new ArrayList<double[]>();
		hiddenNodeBias = new ArrayList<Double>();
		//now we need to assign random weights and biases to each hidden node
		outputNodeWeights = new double[hiddenNodes];
		for (int i = 0; i <hiddenNodes;i++){
			double [] randWeights = new double[5];
			for (int j = 0; j < randWeights.length;j++){
				randWeights[j]= getRandomDouble(-2/5,2/5);
			}
			hiddenNodeWeights.add(randWeights);
			hiddenNodeBias.add(getRandomDouble(-2/5,2/5));
			//since there will be as many output node weights as there are hidden nodes we can assign them here
			outputNodeWeights[i] = getRandomDouble(-1,1);
		}
		outputNodeBias = getRandomDouble(-1,1);
		
		int counter = 1;
		
		for (int i = 0; i < epochs; i++){
			double MSEsum = 0;
			for (int j = 0; j < testData.size(); j++){
				
				//step 2 is to select a data point
				String[] testDataPoint = testData.get(j).split(",");
				double [] dataPoint = {Double.parseDouble(testDataPoint[1]),Double.parseDouble(testDataPoint[2]),Double.parseDouble(testDataPoint[3]),Double.parseDouble(testDataPoint[4]),Double.parseDouble(testDataPoint[5])};
				double cValue = Double.parseDouble(testDataPoint[6]);
				hiddenNode HN = new hiddenNode(Double.parseDouble(testDataPoint[1]),Double.parseDouble(testDataPoint[2]),Double.parseDouble(testDataPoint[3]),Double.parseDouble(testDataPoint[4]),Double.parseDouble(testDataPoint[5]));
				
				//step 3 is to do a forward pass
				HNUvalues = HN.forwardPass(dataPoint, hiddenNodes, hiddenNodeWeights, hiddenNodeBias);
				outputNode ON = new outputNode(HNUvalues[0], HNUvalues[1]);
				ONUvalue = ON.forwardPass(dataPoint, hiddenNodes, HNUvalues, outputNodeWeights, outputNodeBias);
				
				//step 4 is to make a backward pass
				ODeltaValue = ON.backwardPass(cValue, ONUvalue);
				HDeltaValues = HN.backwardPass(cValue, hiddenNodes, outputNodeWeights, ODeltaValue, HNUvalues);
				
				
				//step 5 is to update the weights
				//we will update the weights for the hidden nodes first
				for (int k = 0; k < hiddenNodes; k++){
					double [] newWeights = new double [5];
					double newBias;
					for (int q =0; q <newWeights.length;q++){
						//creating the updated weights
						newWeights[q] = hiddenNodeWeights.get(k)[q] + p*HDeltaValues[k]*HNUvalues[k];
					}
					//changing the weights
					hiddenNodeWeights.set(k, newWeights);
					//now for the bias
					newBias = hiddenNodeBias.get(k) + p*HDeltaValues[k];
					hiddenNodeBias.set(k, newBias);
				}
				
				//now we need to  update the weights for the output node
				for (int t = 0;t <hiddenNodes;t++){
					double oldWeight = outputNodeWeights[t];
					double newWeight = outputNodeWeights[t] + p*ODeltaValue*ONUvalue;
					double deltaWeight = newWeight - oldWeight;
					newWeight = outputNodeWeights[t] + p*ODeltaValue*ONUvalue + 0.9*deltaWeight;
					outputNodeWeights[t] = newWeight;
				}
				//finally we need to update the output node bias. 
				outputNodeBias = outputNodeBias + p*ODeltaValue;
				
				//calculating the mse
				MSEsum = MSEsum + Math.pow((ONUvalue - cValue),2);
				
			}
			double msError = 0;
			if (counter%1000 == 0){
				//read the validation set
				String filePath1 = new File("").getAbsolutePath();
				ArrayList<String> testData1 = importData(filePath + "\\" + "validationSet.txt");
				//now I need to loop through the validation set once.
				double mseSum = 0;
				for (int l = 0; l < testData1.size();l++){
					String[] testDataArray = testData.get(l).split(",");
					//getting the input data
					double[] TD = {Double.parseDouble(testDataArray[1]),Double.parseDouble(testDataArray[2]),Double.parseDouble(testDataArray[3]),Double.parseDouble(testDataArray[4]),Double.parseDouble(testDataArray[5])};
					//getting the correct value
					double cValue = Double.parseDouble(testDataArray[6]); 
					//creating our hidden node class
					hiddenNode HN = new hiddenNode(TD[0],TD[1],TD[2],TD[3],TD[4]);
					//the assosiated U values
					double[] THNUvalues = HN.forwardPass(TD, hiddenNodes, hiddenNodeWeights, hiddenNodeBias);
					//creating our output node
					outputNode ON = new outputNode(THNUvalues[0],THNUvalues[1]);
					//calculating the u value
					double TONUvalues = ON.forwardPass(TD, hiddenNodes, THNUvalues, outputNodeWeights, outputNodeBias);
					//generating the mean squared sum
					mseSum = mseSum + Math.pow((TONUvalues - cValue), 2);
				}
				//calculating the RMSE
				double rmsError = Math.sqrt(mseSum/testData1.size());
				validationError.add(rmsError);
				//comparing the error to the previous, if it is increasing then it will break the loop
				if (validationError.get(validationError.size()-1) > validationError.get(validationError.size()-2) ){
					System.out.println(counter);
					break;
				}
				
			}
			
			
			
			
			counter = counter + 1;
			double RMSE = Math.sqrt(MSEsum/717);
			System.out.println(RMSE);
		}
		
		
		
		//producing a test set to compare against the real values:
		String filePath1 = new File("").getAbsolutePath();
		ArrayList<String> testresults = importData(filePath1 + "\\" + "testData.txt");
		String results = "";
		for (int z = 0; z < testresults.size();z++){
			String[] data = testresults.get(z).split(",");
			double[] resultData = {Double.parseDouble(data[1]),Double.parseDouble(data[2]),Double.parseDouble(data[3]),Double.parseDouble(data[4]),Double.parseDouble(data[5])};
			hiddenNode hiddenN = new hiddenNode(resultData[0],resultData[1],resultData[2],resultData[3],resultData[4]);
			double[] TestHiddenNodeValue = hiddenN.forwardPass(resultData, hiddenNodes, hiddenNodeWeights, hiddenNodeBias);
			outputNode outputN = new outputNode(TestHiddenNodeValue[0],TestHiddenNodeValue[1]);
			double TheOutputNodeU_Values = outputN.forwardPass(resultData, hiddenNodes, TestHiddenNodeValue, outputNodeWeights, outputNodeBias);
			results = results + Double.toString(TheOutputNodeU_Values) + "\n";
		}
		System.out.println("\n");
		System.out.println("\n");
		System.out.println("\n");
		System.out.println(results);
	}
	
	public static ArrayList<String> importData(String file) throws IOException{
		/*
		 * This function takes an input which is the name of a file then
		 * imports the data from the text file and creates a list
		 * then returns the list
		 * 
		 */
		
		//create a list to store the data in 
		ArrayList<String> testData = new ArrayList<String>();
		//create a buffer reader to read the file
		BufferedReader br = new BufferedReader(new FileReader(file));
		//loop through the file adding each line to the list
		String tempStr;
		while ((tempStr = br.readLine())!=null){
			testData.add(tempStr);
		}
		
		return testData;
	}
	
	private static double getRandomDouble(double min, double max){
	    double x = (Math.random()*((max-min)+1))+min;
	    return x;
	}
	
	
	
	

}
