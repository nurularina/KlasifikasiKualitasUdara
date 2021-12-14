function submitform() {
    var f = document.getElementsByTagName('form')[0];
    document.getElementById("box").style.visibility = visible
    if(f.checkValidity()) {
      
      f.submit();
      
    } else {
      alert(document.getElementById('example').validationMessage);
    }
}