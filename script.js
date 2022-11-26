function predict(){



  let val = document.getElementById("myText").value;
  console.log(val);
  let data = {"text": val};

  fetch("http://127.0.0.1:12345/classify", {
  method: "POST",
  mode: 'cors',
  headers: {'Content-Type': 'application/json'}, 
  body: JSON.stringify(data)
  // }).then(res => console.log(res.json()));
  }).then(res => res.json()).then(
    (myBlob) => {
      console.log("result is ", myBlob["classes"], myBlob["classes"].length);
      if(myBlob["classes"].length === 0){
        document.getElementById("result").innerHTML = "No class has higher confidence for this input!";
      }
      else{
        const currentDiv = document.getElementById("result");
        currentDiv.innerHTML = '';

        var i = 0;  
        for (i = 0; i < myBlob["classes"].length; i++) {  
          var newDiv = document.createElement('div');
          var j = 0;

          newDiv.innerHTML += myBlob["classes"][i] + ": ";
          for(j = 0; j < myBlob["class_colors"][i].length; j++){
            newDiv.innerHTML += myBlob["class_colors"][i][j] + " ";
          }
          
          currentDiv.appendChild(newDiv);

          newDiv = document.createElement('div');
          j = 0;

          newDiv.innerHTML += "<br><b>Using Lime<b><br>"
          newDiv.innerHTML += myBlob["classes"][i] + ": ";
          for(j = 0; j < myBlob["lime_colors"][i].length; j++){
            newDiv.innerHTML += myBlob["lime_colors"][i][j] + " ";
          }
          
          currentDiv.appendChild(newDiv);
         }

        var newDiv = document.createElement('div');
        newDiv.innerHTML += "<br>Class Importances : ";

        var j = 0;
        for(j = 0; j < 28; j++){
          if(j == 27){
            newDiv.innerHTML += myBlob["class_attrs"][j] + ".";
          }
          else{
            newDiv.innerHTML += myBlob["class_attrs"][j] + ",";
          }
          
        }
        
        currentDiv.appendChild(newDiv);
      }
      


    }
  );
}