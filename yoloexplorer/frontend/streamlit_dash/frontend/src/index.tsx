import { Streamlit, RenderData } from "streamlit-component-lib"

const labelDiv = document.body.appendChild(document.createElement("label"))
const label = labelDiv.appendChild(document.createTextNode(""))
const container = document.body.appendChild(document.createElement("div"))
container.classList.add("container")
const selected_component_values: number[] = []

/**
 * The component's render function. This will be called immediately after
 * the component is initially loaded, and then again every time the
 * component gets new data from Python.
 */
function onRender(event: Event): void {
  // Get the RenderData from the event
  const data = (event as CustomEvent<RenderData>).detail

  if (data.theme) {
    labelDiv.style.font = data.theme.font
    labelDiv.style.color = data.theme.textColor
    if (data.theme.base === "dark") {
      document.body.querySelectorAll(".box, .caption").forEach((el) => {
        el.classList.add("dark")
      })
    } else {
      document.body.querySelectorAll(".box, .caption").forEach((el) => {
        el.classList.remove("dark")
      })
    }

    // TODO: Gray out the component if it's disabled.
  }

  label.textContent = data.args["label"]
  let images = data.args["images"]
  let captions = data.args["captions"]
  // console.log(captions)

  if (container.childNodes.length === 0) {
    images.forEach((image: string, i: number) => {
      let item = container.appendChild(document.createElement("div"))
      item.classList.add("item")
      if (data.args["use_container_width"] === true) {
        item.classList.add("stretch")
      }

      let box = item.appendChild(document.createElement("div"))
      box.classList.add("image-box")

      let img = box.appendChild(document.createElement("img"))
      img.classList.add("image")
      img.src = image

      if (captions) {
        let caption = item.appendChild(document.createElement("div"))
        caption.classList.add("caption")
        caption.textContent = captions[i]
      }

      // check if i is in the index array
      if (data.args["indices"] !== undefined && data.args["indices"].includes(i)) {
        box.classList.add("selected")
        img.classList.add("selected")
        selected_component_values.push(i)
      }

      img.onclick = function () {
        // check if the image is already selected, then un-select it and remove it from the array
        if (box.classList.contains("selected")) {
          selected_component_values.splice(selected_component_values.indexOf(i), 1)
          box.classList.remove("selected")
          img.classList.remove("selected")
        } else {
          selected_component_values.push(i)
          box.classList.add("selected")
          img.classList.add("selected")
        }
        selected_component_values.sort()
        Streamlit.setComponentValue(selected_component_values)
      }
    })
    // return selected_component_values
    Streamlit.setComponentValue(selected_component_values)
  }

  // We tell Streamlit to update our frameHeight after each render event, in
  // case it has changed. (This isn't strictly necessary for the example
  // because our height stays fixed, but this is a low-cost function, so
  // there's no harm in doing it redundantly.)
  Streamlit.setFrameHeight()
}

// Attach our `onRender` handler to Streamlit's render event.
Streamlit.events.addEventListener(Streamlit.RENDER_EVENT, onRender)

// Tell Streamlit we're ready to start receiving data. We won't get our
// first RENDER_EVENT until we call this function.
Streamlit.setComponentReady()

// Finally, tell Streamlit to update our initial height. We omit the
// `height` parameter here to have it default to our scrollHeight.
Streamlit.setFrameHeight()
