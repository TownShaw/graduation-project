<template>
  <div id="inputFiles">
    <h1 style="font-family: inherit">Educational Video Knowledge Prediction</h1>
    <div id="knowledge-system" class="select-knowledge">
      <!-- <label class="mr-sm-2" for="inlineFormCustomSelect">Knowledge System</label> -->
      <select id="know-selector" class="custom-select mr-sm-2" @change="handleFileUpLoad($event, 'datasetName')">
        <option selected>Please select the Knowledge System...</option>
        <option value="Khan">Khan</option>
      </select>
    </div>
    <br />
    <label class="custom-file-upload">
      <input type="file" name="video" @change="handleFileUpLoad($event, 'video')" class="">
      <span id="video-title" class="span-text">Video upload</span>
      <br />
      <span id="video-filename">{{ this.videoFileName }}</span>
    </label>
    <label class="custom-file-upload">
      <input type="file" name="text" @change="handleFileUpLoad($event, 'subtitle')" class="">
      <span id="subtitle-title" class="span-text">Subtitle upload</span>
      <br />
      <span id="subtitle-filename">{{ this.subtitleFileName }}</span>
    </label>
    <br />
    <br />
    <input type="submit" name="submit" @click="submitFiles()" class="btn btn-primary mb-2">
  </div>
</template>

<script>
import { postRequest } from '../utils/api'

export default {
  name: "VideoPost",
  emits: ["resultChange"],
  data() {
    return {
      datasetName: undefined,
      subtitle: undefined,
      video: undefined
    }
  },
  methods: {
    handleFileUpLoad(event, fileKind) {
      if (fileKind === "datasetName") {
        this.datasetName = event.target.value;
      } else if (fileKind === 'video') {
        this.video = event.target.files[0];
      } else if (fileKind === "subtitle") {
        this.subtitle = event.target.files[0];
      }
    },
    submitFiles() {
      if (!this.datasetName) {
        alert("You MUST select a Knowledge System!");
        return;
      } else if (!this.video) {
        alert("You MUST upload a video file!");
        return;
      } else if (!this.subtitle) {
        alert("You MUST upload a subtitle file of the video you have uploaded!");
        return;
      }
      const formData = new FormData();
      formData.append('datasetName', this.datasetName);
      formData.append('video', this.video);
      formData.append('subtitle', this.subtitle);
      let headers = {
            // 'Origin':'http://localhost:4000',
            "Content-Type": "multipart/form-data"
            // "Content-Type": "application/json"
        }
      // postRequest('', {"video": this.video, "subtitle": this.subtitle}).then((res) => {
      postRequest('', formData, headers).then((resp) => {
        if (resp.status == 200) {
          let jsonData = resp.data;   // predict results
          this.$emit("resultChange", jsonData);  // emit results to parent component
        }
      }).catch(() => {
        this.$emit("resultChange", {})
      });
      // clear local variables
      // this.subtitle = undefined
      // this.video = undefined
    }
  },
  computed: {
    subtitleFileName() {
      if (this.subtitle) {
        return this.subtitle.name
      }
      else {
        return ""
      }
    },
    videoFileName() {
      if (this.video) {
        return this.video.name
      }
      else {
        return ""
      }
    }
  }
}
</script>

<style>
h1 {
  color: rgba(0, 0, 0, 0.7);
}

#know-selector {
  margin: 15px;
}

.select-knowledge {
  margin-left: 12%;
  margin-right: 12%;
}

input[type="file"] {
  display: none;
}

.custom-file-upload {
  border: 1px solid #ccc;
  display: inline-block;
  padding: 6px 12px;
  cursor: pointer;
  margin-left: 10px;
  margin-right: 10px;
  background-color: rgba(233, 233, 233, 90%);
  background-clip: padding-box;
}

.custom-file-upload:hover {
    border: 1px solid rgb(56, 56, 56);
}

.span-text {
  font-weight: bold;
  color: rgba(61, 61, 61, 0.9);
}
</style>
