package titanic

import titanic.NaiveBayes.{calcAttribValuesForEachClass, calcClassValuesForPrediction, calcConditionalPropabilitiesForEachClass, calcPriorPropabilities, countAttributeValues, findBestFittingClass}

object TitanicDataSet {

  /**
   * Creates a model that predicts 1 (survived) if the person of the certain record
   * is female and 0 (deceased) otherwise
   *
   * @return The model represented as a function
   */
  def simpleModel:(Map[String, Any], String) => (Any, Any)= {
    (map,id_key) => (map(id_key), if(map("sex")=="female") 1 else 0)
  }

  /**
   * This function should count for a given attribute list, how often an attribute is
   * not present in the data records of the data set
   *
   * @param data    The DataSet where the counting takes place
   * @param attList List of attributes where the missings should be counted
   * @return A Map that contains the attribute names (key) and the number of missings (value)
   */
  def countAllMissingValues(data: List[Map[String, Any]], attList: List[String]): Map[String, Int] = {
//    println(attList.map(att => data.map(entry => entry(att))))
    val resMap = scala.collection.mutable.Map[String, Int]()
    for (entry <- data) yield {
      attList.filterNot(entry.keys.toList.contains(_)).map(value => resMap.update(value, resMap.getOrElse(value, 0) + 1))
    }
    resMap.toMap
  }

  /**
   * This function should extract a set of given attributes from a record
   *
   * @param record  Record that should be extracted
   * @param attList List of attributes that should be extracted
   * @return A Map that contains only the attributes that should be extracted
   *
   */
    // https://alvinalexander.com/scala/how-to-filter-map-filterkeys-retain-scala-cookbook/
  def extractTrainingAttributes(record:Map[String, Any], attList:List[String]):Map[String, Any]= {
    record.filterKeys(attList.contains)
  }

  /**
   * This function should create the training data set. It extracts the necessary attributes,
   * categorize them and deals with the missing values. You could find some hints in the description
   * and the lectures
   *
   * @param data Training Data Set that needs to be prepared
   * @return Prepared Data Set for using it with Naive Bayes
   */
  def createDataSetForTraining(data:List[Map[String, Any]]): List[Map[String, Any]] = {
//    val validAges1 = data.map(x => x.getOrElse("age", -1)).filterNot(_ == -1).map(_.toString.toFloat)
//    val avgAge1 = validAges1.sum / validAges1.size
    val attList = List("passengerID", "sex", "age", "survived", "pclass")
    val sumAges = data.map(entry => entry.getOrElse("age", -1)).filter(e => e != -1).map(e => (e, 1)).reduce((x1, x2) => ((x1._1.asInstanceOf[Float] + x2._1.asInstanceOf[Float]), x1._2 + x2._2))
    val avgAge = sumAges._1.asInstanceOf[Float] / sumAges._2

//    println(data.map(elem => extractTrainingAttributes(elem, attList)).map(y => y.updated("age", rateAge(y.getOrElse("age", avgAge)))))
    data.map(elem => extractTrainingAttributes(elem, attList)).map(y => y.updated("age", rateAge(y.getOrElse("age", avgAge))))
  }
  def rateAge(ageAny: Any): Int = {
    val age = ageAny.toString.toFloat
    if (age < 15) 0
    else if (age < 45) 1
    else 2
  }

  /**
   * This function builds the model. It is represented as a function that maps a data record
   * and the name of the id-attribute to the value of the id attribute and the predicted class
   * (similar to the model building process in the train example)
   *
   * @param trainDataSet  Training Data Set
   * @param classAttrib name of the attribute that contains the class
   * @return A tuple consisting of the id (first element) and the predicted class (second element)
   */
  def createModelWithTitanicTrainingData(tdata:List[Map[String,Any]], classAttr:String):
     (Map[String, Any], String) => (Any, Any)= {

        val trainData = createDataSetForTraining(tdata)
        val classVals= countAttributeValues(trainData,classAttr)
        val data= calcAttribValuesForEachClass(trainData,classAttr)
        val condProp = calcConditionalPropabilitiesForEachClass(data,classVals)
        val prior= calcPriorPropabilities(trainData,classAttr)
        (map,id_key) => (map(id_key),findBestFittingClass(calcClassValuesForPrediction(map-id_key,condProp,prior)))

  }
}
