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
        //设置参数 
        lmp = new LogisticModelParameters();  
        output = new PrintWriter(new OutputStreamWriter(System.out,  
                Charsets.UTF_8), true);  
        lmp.setLambda(0.001);  
        lmp.setLearningRate(50);  
        lmp.setMaxTargetCategories(4); //总共有4列特征值  
        lmp.setNumFeatures(2);         //预测结果只有0和1两种
        List<String> targetCategories = Lists.newArrayList("DayofMonth", "DayOfWeek", "FlightNum","Distance");  //特征值
        lmp.setTargetCategories(targetCategories);  
        lmp.setTargetVariable("ArrDelay"); // 需要进行预测的是ArrDelay属性  
        List<String> typeList = Lists.newArrayList("numeric", "numeric", "numeric", "numeric");  
        List<String> predictorList = Lists.newArrayList("sepallength", "sepalwidth", "petallength", "petalwidth");  
        lmp.setTypeMap(predictorList, typeList);  
        //读数据  
        List<String> raw = FileUtils.readLines(new File("/home/hadoop/桌面/大数据三级项目/清洗后的数据/air1998ForPre.csv"), "UTF-8"); //使用common-io进行文件读取  
        System.out.println(FileUtils.readLines(new File("/home/hadoop/桌面/大数据三级项目/清洗后的数据/air1998ForPre.csv"), "UTF-8")); 
        String header = raw.get(0);  
        List<String> content = raw.subList(1, raw.size());  
        // parse data  
        CsvRecordFactory csv = lmp.getCsvRecordFactory();  
//        csv.firstLine(header); 
        
        //训练 
        OnlineLogisticRegression lr = lmp.createRegression();  
            for (String line : content) {  
                Vector input = new RandomAccessSparseVector(lmp.getNumFeatures());  
                int targetValue = csv.processLine(line, input);  
                lr.train(targetValue, input);
            }  
        //输出准确率
        double correctRate = 0;  
        double sampleCount = content.size();  
          
        for (String line : content) {  
            Vector v = new SequentialAccessSparseVector(lmp.getNumFeatures());  
            int target = csv.processLine(line, v);  
            int score = lr.classifyFull(v).maxValueIndex();  // 分类核心语句!!!  
            if(score == target) {  
                correctRate++;  
            }  
        }  
        output.printf(Locale.ENGLISH, "Rate = %.2f%n", correctRate / sampleCount);  
    }  
  
}  