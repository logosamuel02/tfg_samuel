async function set_loaded_figure(fig, title, plot){
    $.get(`http://localhost:10000/manager/plots/anova/${fig}`, function(data) {
    $(`#${title}`).html(data[0])
    $(`#${plot}`).html(data[1])
  }).catch((error) => {
  $(`#${plot}`).html(error);
  });
  console.log("GET CALL")
}

async function reload(){
  if (document.getElementById('reload_cb').checked) 
  {
      $('#reload_input').prop('disabled', true);
      if (typeof myInterval !== 'undefined') {
        clearInterval(myInterval)
      }
      console.log("Reload ON");
      set_loaded_figure("table")
      window.myInterval = setInterval(function() { set_loaded_figure("table"); }, Number(document.getElementById("reload_input").value)*1000);

  }
  else {
    $('#reload_input').prop('disabled', false);
    console.log("Reload OFF")
    if (typeof myInterval !== 'undefined') {
        clearInterval(myInterval)
      };
  }
}

async function regenerate(){
  if (document.getElementById('regen_cb').checked) 
  {
      $('#regen_input').prop('disabled', true);
      if (typeof myInterval2 !== 'undefined') {
        clearInterval(myInterval2)
      }
      console.log("Regenerate ON");
      set_generate_figure("figure")
      window.myInterval2 = setInterval(function() { set_generate_figure("figure"); }, Number(document.getElementById("reload_input").value)*1000);

  }
  else {
    $('#regen_input').prop('disabled', false);
    console.log("Regenerate OFF")
    if (typeof myInterval2 !== 'undefined') {
        clearInterval(myInterval2)
      };
  }
}

async function set_form_arguments(args, form){
    $.get(`http://localhost:10000/manager/plots/anova/${args}`, function(data) {
    console.log("Finished: get args to generate")
     var f = document.getElementById(form)
    //Create and append the options
    for (const parent in data){
      let textNode = document.createTextNode(parent); 
      f.appendChild(textNode);
      var selectList = document.createElement("select");
      selectList.id = parent;
      selectList.form = f.id
      selectList.name = parent

      for (const child in data[parent]){
        var option = document.createElement("option");
        option.value = data[parent][child];
        option.text = data[parent][child];
        selectList.appendChild(option)
      }
      f.appendChild(selectList)
      linebreak = document.createElement("br");
      f.appendChild(linebreak);
    }
    var s = document.createElement("input"); //input element, Submit button
    s.setAttribute('type',"submit");
    s.setAttribute('id',"sub");
    s.setAttribute('value',"Generate");
    f.appendChild(s)
  }).catch((error) => {
  console.log(error)
  });
  console.log("Start: get args to generate")
}

async function set_generate_figure(fig, form, title, plot){
    $.post(`http://localhost:10000/manager/plots/anova/${fig}`, $(`#${form}`).serialize(), function(data) {
      console.log("Finished: set_generate_figure to generate")
    $(`#${title}`).html(data[0])
    $(`#${plot}`).html(data[1])
  }).catch((error) => {
  $(`#${plot}`).html("Any of the selected attributes is invalid.");
  });
  console.log("Start: set_generate_figure to generate")
}

async function form_sender(form, figure) {
    form = document.querySelector(`#${form}`);
    form.addEventListener("submit", (event) => {
        event.preventDefault();
        set_generate_figure(figure);
        });   
}

window.onload = function() {
var num = document.querySelector("#reload_input")
var span = document.querySelector("#sl1")
num.addEventListener("input", function() {
  if (num.value >= 1 && num.value <= 100000){
    $('#reload_cb').prop('disabled', false);
    // span.style.background='#ccc';
  }
  else {
    $('#reload_cb').prop('disabled', true);
    // span.style.background="#FF370F";
  }
});
};  

console.log()

set_loaded_figure("table", "title", "figure")
set_generate_figure("figure", "form1", "title", "figure")
set_form_arguments("args_table", "form1")
form_sender("form1", "figure")
