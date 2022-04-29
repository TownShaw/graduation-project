<template>
  <div v-if="sections && sections.length != 0">
    <!-- <p>{{ this.video_labels }}</p> -->
    <!-- <li v-for="idx of section_labels.length" :key="idx - 1">{{ sections[idx - 1] }} ~ {{ sections[idx] }}: {{ section_labels[idx - 1] }}</li> -->
    <h2 style="text-align: left">Global Labels:</h2>
    <table id="global-labels" class="table table-bordered table-margin">
      <td v-for="idx of video_labels.length" :key="idx - 1"
          :style="selectStyleByLevel(video_labels[idx - 1][1])">{{ video_labels[idx - 1][0] }}</td>
    </table>
    <h2 style="text-align: left">Sections Labels:</h2>
    <table id="section-labels" class="table table-bordered table-margin">
      <tr>
        <th class="table-info" v-for="idx of section_labels.length" :key="idx - 1">{{ sections[idx - 1] }} ~ {{ sections[idx] }}</th>
      </tr>
      <tr class="table-secondary" v-for="idx of maxSectionLabelLength" :key="idx - 1">
        <template v-for="section_idx of section_labels.length" :key="section_idx - 1">
          <td class="td-border" v-if="idx - 1 < section_labels[section_idx - 1].length"
              :style="selectStyleByLevel(section_labels[section_idx - 1][idx - 1][1])">{{ section_labels[section_idx - 1][idx - 1][0] }}</td>
          <td class="td-border" v-else></td>
        </template>
      </tr>
    </table>
  </div>
</template>

<script>
export default {
  name: "ResultDisplay",
  props: {
    jsonData: {
      type: Object
    }
  },
  data () {
    return {
      sections: undefined,
      section_labels: undefined,
      video_labels: undefined
    }
  },
  methods: {
    selectStyleByLevel(level) {
      var classes = {
        0: {
          "background-color": "rgba(255, 132, 61, 0.7)",
          "background-clip": "padding-box"
        },
        1: {
          "background-color": "rgba(255, 251, 20, 0.7)",
          "background-clip": "padding-box"
        },
        2: {
          "background-color": "rgba(60, 152, 204, 0.7)",
          "background-clip": "padding-box"
        }
      };
      return classes[level];
    }
  },
  computed: {
    maxSectionLabelLength() {
      var maxLen = 0;
      for (const labels of this.section_labels) {
        maxLen = Math.max(maxLen, labels.length);
      }
      return Math.max(maxLen, 1);
    }
  },
  beforeUpdate() {
    this.sections = this.jsonData.sections
    this.section_labels = this.jsonData.section_labels
    this.video_labels = this.jsonData.video_labels
  },
}
</script>

<style>
li {
  text-align: left;
}

.table-margin {
  margin-bottom: 3%;
}

table.table-bordered {
  border: 2px rgba(58, 58, 58, 0.5);
  border-style: dotted;
  border-collapse: collapse;
}

table.table-bordered th {
  color: rgb(43, 43, 43);
  border-left: 2px solid rgba(43, 43, 43, 0.5);
  border-right: 2px solid rgba(43, 43, 43, 0.5);
  background-color: rgba(21, 221, 211, 0.7);
  background-clip: padding-box;
}

table th:first-child {
    border-left: none;
}

table th:last-child {
    border-right: none;
}

table.table-bordered tr {
  background-color: rgba(226, 226, 226, 0.7);
  background-clip: padding-box;
}

table.table-bordered td {
  border-left: 2px solid rgba(43, 43, 43, 0.5);
  border-right: 2px solid rgba(43, 43, 43, 0.5);
  background-color: rgba(230, 230, 230, 0.6);
  background-clip: padding-box;
}

table td:first-child {
    border-left: none;
}

table td:last-child {
    border-right: none;
}
</style>