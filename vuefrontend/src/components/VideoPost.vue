<template>
  <div id="inputFiles">
    <p>Post Data to Backend:</p>
    <div id="knowledge-system">
      <label class="mr-sm-2" for="inlineFormCustomSelect">Knowledge System</label>
      <select id="know-selector" class="custom-select mr-sm-2" @change="handleFileUpLoad($event, 'datasetName')">
        <option selected>Choose...</option>
        <option value="Khan">Khan</option>
      </select>
    </div>
    <label class="custom-file-upload">
      <input type="file" name="video" @change="handleFileUpLoad($event, 'video')" class="">
      <span id="video-title" style="font-weight: bold">Video upload</span>
      <br />
      <span id="video-filename">{{ this.videoFileName }}</span>
    </label>
    <label class="custom-file-upload">
      <input type="file" name="text" @change="handleFileUpLoad($event, 'subtitle')" class="">
      <span id="subtitle-title" style="font-weight: bold">Subtitle upload</span>
      <br />
      <span id="subtitle-filename">{{ this.subtitleFileName }}</span>
    </label>
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
        this.datasetName = event.target.value
      } else if (fileKind === 'video') {
        this.video = event.target.files[0]
      } else if (fileKind === "subtitle") {
        this.subtitle = event.target.files[0]
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
      this.subtitle = undefined
      this.video = undefined
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
#know-selector {
  margin: 15px;
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
}
.custom-file-upload:hover {
    border: 1px solid rgb(56, 56, 56);
}
</style>
