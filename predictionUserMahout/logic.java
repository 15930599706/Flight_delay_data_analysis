package com.mine.mahout.practice;  
  
import java.io.File;  
import java.io.IOException;  
import java.io.OutputStreamWriter;  
import java.io.PrintWriter;  
import java.util.List;  
import java.util.Locale;  
  
import org.apache.commons.io.FileUtils;  
import org.apache.mahout.classifier.evaluation.Auc;
import org.apache.mahout.classifier.sgd.CsvRecordFactory;
import org.apache.mahout.classifier.sgd.OnlineLogisticRegression;
import org.apache.mahout.math.RandomAccessSparseVector;  
import org.apache.mahout.math.SequentialAccessSparseVector;  
import org.apache.mahout.math.Vector;  
  
import com.google.common.base.Charsets;  
import com.google.common.collect.Lists;  
  
public class test {  
  
    private static LogisticModelParameters lmp;  
    private static PrintWriter output;  

	public static void main(String[] args) throws IOException {  
        //���ò��� 
        lmp = new LogisticModelParameters();  
        output = new PrintWriter(new OutputStreamWriter(System.out,  
                Charsets.UTF_8), true);  
        lmp.setLambda(0.001);  
        lmp.setLearningRate(50);  
        lmp.setMaxTargetCategories(4); //�ܹ���4������ֵ  
        lmp.setNumFeatures(2);         //Ԥ����ֻ��0��1����
        List<String> targetCategories = Lists.newArrayList("DayofMonth", "DayOfWeek", "FlightNum","Distance");  //����ֵ
        lmp.setTargetCategories(targetCategories);  
        lmp.setTargetVariable("ArrDelay"); // ��Ҫ����Ԥ�����ArrDelay����  
        List<String> typeList = Lists.newArrayList("numeric", "numeric", "numeric", "numeric");  
        List<String> predictorList = Lists.newArrayList("sepallength", "sepalwidth", "petallength", "petalwidth");  
        lmp.setTypeMap(predictorList, typeList);  
        //������  
        List<String> raw = FileUtils.readLines(new File("/home/hadoop/����/������������Ŀ/��ϴ�������/air1998ForPre.csv"), "UTF-8"); //ʹ��common-io�����ļ���ȡ  
        System.out.println(FileUtils.readLines(new File("/home/hadoop/����/������������Ŀ/��ϴ�������/air1998ForPre.csv"), "UTF-8")); 
        String header = raw.get(0);  
        List<String> content = raw.subList(1, raw.size());  
        // parse data  
        CsvRecordFactory csv = lmp.getCsvRecordFactory();  
//        csv.firstLine(header); 
        
        //ѵ�� 
        OnlineLogisticRegression lr = lmp.createRegression();  
            for (String line : content) {  
                Vector input = new RandomAccessSparseVector(lmp.getNumFeatures());  
                int targetValue = csv.processLine(line, input);  
                lr.train(targetValue, input);
            }  
        //���׼ȷ��
        double correctRate = 0;  
        double sampleCount = content.size();  
          
        for (String line : content) {  
            Vector v = new SequentialAccessSparseVector(lmp.getNumFeatures());  
            int target = csv.processLine(line, v);  
            int score = lr.classifyFull(v).maxValueIndex();  // ����������!!!  
            if(score == target) {  
                correctRate++;  
            }  
        }  
        output.printf(Locale.ENGLISH, "Rate = %.2f%n", correctRate / sampleCount);  
    }  
  
}  