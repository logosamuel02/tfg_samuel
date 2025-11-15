async function load_all_figures() {
  for (var plot_string of window.plots){
    load_figure(plot_string)
}
}

async function generate_all_figures() {
  for (var plot_string of window.plots){
    generate_figure(plot_string)
}
}

async function reload(){
  if (document.getElementById('reload_cb').checked) 
  {
      $('#reload_input').prop('disabled', true);
      if (typeof myInterval !== 'undefined') {
        clearInterval(myInterval)
      }
      console.log("Reload ON");
      load_all_figures()
      window.myInterval = setInterval(function() { load_all_figures(); }, Number(document.getElementById("reload_input").value)*1000);

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
      generate_all_figures()
      window.myInterval2 = setInterval(function() { generate_all_figures(); }, Number(document.getElementById("reload_input").value)*1000);

  }
  else {
    $('#regen_input').prop('disabled', false);
    console.log("Regenerate OFF")
    if (typeof myInterval2 !== 'undefined') {
        clearInterval(myInterval2)
      };
  }
}

async function load_figure(plot){
    $.post(`http://localhost:10000/manager/plots/anova/get_figure`, plot, function(data) {
    // console.log(`Finish: get_${plot}`)
    $(`#title_${plot}`).html(data[0])
    $(`#figure_${plot}`).html(data[1])
  }).catch((error) => {
  // $(`#figure_${plot}`).html(error);
  console.log(error)
  });
  // console.log(`Start: get_${plot}`)
}

async function load_arguments(plot){
    $.post(`http://localhost:10000/manager/plots/anova/get_args`, plot, function(data) {
    // console.log(`Finished: args_${plot}`)
     var f = document.getElementById(`form_${plot}`)
     var input = document.createElement("select");
      input.hidden = "hidden";
      input.name = "name"
      input.form = f.id
      input.id = `hidden_${plot}`
      var option = document.createElement("option");
        option.value = plot
        option.text = plot
        input.appendChild(option)
      f.appendChild(input)
    //Create and append the options
    for (const parent in data){
      let textNode = document.createTextNode(parent); 
      f.appendChild(textNode);
      var selectList = document.createElement("select");
      selectList.id = `${parent}_${plot}`;
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
    s.setAttribute('id',`sub_${plot}`);
    s.setAttribute('value',"Generate");
    f.appendChild(s)
  }).catch((error) => {
  console.log(error)
  });
  // console.log(`Start: args_${plot}`)
}

async function generate_figure(plot){
    $.post(`http://localhost:10000/manager/plots/anova/generate_figure`, $(`#form_${plot}`).serialize(), function(data) {
      // console.log(`Finished: gen_${plot}`)
    $(`#title_${plot}`).html(data[0])
    $(`#figure_${plot}`).html(data[1])
  }).catch((error) => {
  // $(`#figure_${plot}`).html("Any of the selected attributes is invalid.");
    console.log(error)
});
  // console.log(`Start: gen_${plot}`)
}

async function send_form(plot) {
    form = document.querySelector(`#form_${plot}`);
    form.addEventListener("submit", (event) => {
        event.preventDefault();
        generate_figure(plot);
        });   
}

const create_containers = (plots) => {
    var containers = []
  for (const plot of plots){
    console.log(plot)
    let container = "";
	container += `<div id=\"filter-target-${plot}\" class=\"row mt-3\">\n`;
	container += "\t  <div class=\"card\">\n";
	container += `\t\t<h5 class=\"card-header\" id=\"title_${plot}\"></h5>\n`;
	container += "\t\t<div class=\"plotly-chart\">\n";
	container += `\t\t  <form id=\"form_${plot}\" target=\"hiddenFrame\">`;
  container += `<select name=\"name\" hidden=\"hidden\" form=\"form_${plot}\" id=\"hidden_${plot}\">`
  container += `<option value=\"${plot}\"></option>`
  container += "</select>"
  container += "\t\t</form>\n"
	container += "\t\t</div>\n";
	container += "\t\t<div class=\"card-body\">\n";
	container += `\t\t  <div class=\"plotly-chart\" id=\"figure_${plot}\">\n`;
	container += "\t\t  </div>\n";
	container += "\t\t</div>\n";
	container += "\t  </div>\t   \n";
	container += "</div>\n";
  containers.push(container)
  }
  console.log("Finish: created containers")
	return containers.join(" ");
}

console.log(window.plots)

// Create containers
var containers = create_containers(window.plots)
document.getElementById("card_containers").innerHTML = containers;

// Set up initial calls to load and generate figures
for (var plot of window.plots){
  load_figure(plot)
  load_arguments(plot)
  generate_figure(plot)
  send_form(plot)
  console.log(`Calls for ${plot}`)
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
